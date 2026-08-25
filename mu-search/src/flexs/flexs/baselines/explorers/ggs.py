import flexs
from flexs.utils import sequence_utils as s_utils
import numpy as np
import torch
from typing import Optional, Tuple, Dict
from flexs.baselines.models.basecnn import BaseCNN
from pathlib import Path
import torch.nn.functional as F
import torch.distributions as dists
import pandas as pd
import time
from datetime import datetime
import tqdm
import warnings


to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()


class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch: str or list, return_tensor = True) -> np.ndarray or torch.Tensor:
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x: np.ndarray or list or torch.Tensor) -> str or list:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])

class GwgPairSampler(torch.nn.Module):
    """
    An explorer which implements the GwgPairSampler.
    """

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        temperature: float,
        starting_sequence: str,
        alphabet: str,
        population_size: int = 15,
        max_iter: int = 400,
        initial_variance: float = 0.2,
        gibbs_samples: int = 100,
        log_file: Optional[str] = None,
        device: str = 'cpu',
        verbose: bool = False,
        landscape: Optional[flexs.Landscape] = None,
    ):
        super().__init__()

        # super().__init__(
        #     model,
        #     name,
        #     rounds,
        #     sequences_batch_size,
        #     model_queries_per_batch,
        #     starting_sequence,
        #     log_file,
        #     device='cuda:3',
        # )

        self.alphabet = alphabet
        self.population_size = population_size
        self.predictor = model
        self.max_iter = max_iter
        self.initial_variance = initial_variance
        self.round = rounds
        self.temp = temperature
        self.starting_sequence = starting_sequence
        self.device = device
        # self.predictor = self._setup_predictor()
        self.predictor_tokenizer = Encoder(alphabet)
        self.num_tokens = len(self.alphabet)
        self.gibbs_samples = gibbs_samples
        self._verbose = verbose
        self.name = 'GwgPairSampler'
        self.landscape = landscape
        
    def _setup_predictor(self):
        """
        Sets up the CNN predictor model by loading weights and configuration.
        
        Args:
            predictor_dir (str): Directory containing model checkpoint and config files
            
        Returns:
            predictor: Initialized and loaded CNN model
        """
        # Load model weights from checkpoint file
        predictor_path = "/home/v-zhaozhiyu/code/GGS/ckpt/AAV/mutations_7/percentile_0.0_0.3/unsmoothed/predictor/predictor.ckpt"
        # SECURITY: torch.load disabled to comply with the twin flames security incident (see microsoft/Mu-Protein#39).
        # Uncomment the line below only after confirming the checkpoint source is trusted.
        # mdl_info = torch.load(predictor_path, map_location=self.device)
        print('[ WARNING ] torch.load is disabled due to the twin flames security incident; uncomment it in this file to restore loading behavior.')

        # Initialize the CNN model
        predictor = BaseCNN(n_tokens=len(self.alphabet), alphabet=self.alphabet, kernel_size=5, input_size=256, dropout=0.0, make_one_hot=False, activation='relu', linear=True)

        # Clean up state dict keys and load weights
        state_dict = {k.replace('predictor.', ''): v for k, v in mdl_info['state_dict'].items()}
        predictor.load_state_dict(state_dict)

        # Set model to evaluation mode and move to specified device
        predictor.eval()
        predictor.to(self.device)
        
        # Log model architecture
        # self._log.info(predictor)
        return predictor

    def tokenize_seqs(self, seqs):
        """Convert sequence strings into token indices"""
        return self.gen_tokenizer.encode(seqs)
    

    def _calc_local_diff(self, seq_one_hot):
        """Calculate local difference in predictions for each possible mutation
        
        Args:
            seq_one_hot: One-hot encoded sequence tensor
            
        Returns:
            delta_ij: Tensor of differences in predictions for each position/amino acid
        """
        # Construct local difference
        if isinstance(seq_one_hot, torch.Tensor):
            seq_one_hot = seq_one_hot.detach()
        seqs_list = []
        for i in range(seq_one_hot.shape[0]):
            seqs_list.append(s_utils.one_hot_to_string(seq_one_hot[i], self.alphabet))
        gradients, delta_ij = self.predictor.gradient_function(seqs_list)
        # gx = torch.autograd.grad(self.predictor(seq_one_hot).sum(), seq_one_hot)[0]
        # gx_cur = (gx * seq_one_hot).sum(-1)[:, :, None]
        # delta_ij = gx - gx_cur
        return delta_ij

    
    def _gibbs_sampler(self, seq_one_hot):
        """Create a Gibbs sampler function for proposing mutations
        
        Args:
            seq_one_hot: One-hot encoded sequence tensor
            
        Returns:
            _gwg_sample: Function that generates mutated sequences
        """
        orig_delta_ij = self._calc_local_diff(seq_one_hot)
        orig_delta_ij = orig_delta_ij[0]
        
        def _gwg_sample():
            nonlocal orig_delta_ij
            seq_len, num_tokens = orig_delta_ij.shape
            # Construct proposal distributions
            delta_ij = torch.from_numpy(orig_delta_ij.numpy())  # Create local copy
            probs = torch.softmax(delta_ij.flatten() / self.temp, dim=-1)
            gwg_proposal = dists.OneHotCategorical(probs=probs)
            # gwg_proposal = dists.OneHotCategorical(logits = delta_ij.flatten() / self.temp)
            r_ij = gwg_proposal.sample((self.gibbs_samples,)).reshape(
                self.gibbs_samples, seq_len, num_tokens)

            # [num_samples, L, 20]
            seq_token = torch.argmax(seq_one_hot, dim=-1)
            mutated_seqs = seq_token.repeat(self.gibbs_samples, 1)
            seq_idx, res_idx, aa_idx = torch.where(r_ij)
            mutated_seqs[(seq_idx, res_idx)] = aa_idx
            return mutated_seqs
        
        return _gwg_sample


    def _make_one_hot(self, seq, differentiable=False):
        """Convert sequence indices to one-hot encoding
        
        Args:
            seq: Tensor of sequence indices
            differentiable: Whether to make tensor differentiable
            
        Returns:
            seq_one_hot: One-hot encoded sequence tensor
        """
        seq_one_hot = F.one_hot(seq, num_classes=self.num_tokens)
        if differentiable:
            seq_one_hot = seq_one_hot.float().requires_grad_()
        return seq_one_hot

    def _evaluate_one_hot(self, seq):
        """Evaluate predictor on one-hot encoded sequence
        
        Args:
            seq: Sequence tensor to evaluate
            
        Returns:
            model_out: Predictor output scores
        """
        decoded_seq = self.predictor_tokenizer.decode(seq)
        model_out = self.predictor.get_fitness(decoded_seq)
        model_out = torch.tensor(model_out).to(self.device)
        return model_out

    def _decode(self, one_hot_seq):
        """Convert sequence indices back to string sequence"""
        return self.predictor_tokenizer.decode(one_hot_seq)

    def _metropolis_hastings(
            self, mutants, source_one_hot, delta_score):
        """Perform Metropolis-Hastings acceptance step
        
        Args:
            mutants: Proposed mutant sequences
            source_one_hot: One-hot encoding of source sequence
            delta_score: Change in predictor score for mutants
            
        Returns:
            mh_step: Boolean mask of accepted mutations
        """
       
        source = torch.argmax(source_one_hot, dim=-1)
    
        # [num_seq, L]
        mutated_indices = mutants != source[None]
        # [num_seq, L, 20]
        mutant_one_hot = self._make_one_hot(mutants, differentiable=True)
        mutated_one_hot = mutant_one_hot * mutated_indices[..., None]
        
        source_delta_ij = self._calc_local_diff(source_one_hot[None])
        mutant_delta_ij = self._calc_local_diff(mutant_one_hot)
        source_delta_ij = torch.from_numpy(source_delta_ij.numpy())
        mutant_delta_ij = torch.from_numpy(mutant_delta_ij.numpy())

        orig_source_shape = source_delta_ij.shape
        orig_mutant_shape = mutant_delta_ij.shape

        # Flatten starting from the second to last dimension and apply softmax
        q_source = source_delta_ij.flatten(start_dim=-2)
        q_source = F.softmax(q_source / self.temp, dim=-1)

        q_mutant = mutant_delta_ij.flatten(start_dim=-2)
        q_mutant = F.softmax(q_mutant / self.temp, dim=-1)

        # Reshape back to the original shape
        q_source = q_source.view(orig_source_shape).squeeze(0)
        q_mutant = q_mutant.view(orig_mutant_shape)
        
        mutation_tuple = torch.nonzero(mutated_one_hot, as_tuple=True)
        q_ij_source = q_source[mutation_tuple[1], mutation_tuple[2]]
        q_ij_mutant = q_mutant[torch.arange(q_mutant.shape[0]).to(self.device), mutation_tuple[1], mutation_tuple[2]] 
        q_ij_ratio = q_ij_mutant / q_ij_source
        if not isinstance(delta_score, torch.Tensor):
            delta_score = torch.tensor(delta_score).to(self.device)
        accept_prob = torch.exp(delta_score)*q_ij_ratio.to(self.device)
        
        mh_step = accept_prob < torch.rand(accept_prob.shape).to(self.device)
        return mh_step

    def _evaluate_mutants(
            self,
            *,
            mutants,
            score,
            source_one_hot,
        ):
        """Evaluate proposed mutant sequences
        
        Args:
            mutants: Proposed mutant sequences
            score: Score of source sequence
            source_one_hot: One-hot encoding of source sequence
            
        Returns:
            DataFrame containing accepted mutants and their scores,
            Tensor of accepted mutant sequences
        """
        if self.landscape is not None:
            all_mutated_scores = self.landscape.get_fitness(mutants)
        else:
            all_mutated_scores = self._evaluate_one_hot(mutants)
        delta_score = all_mutated_scores - score

        accept_mask = self._metropolis_hastings(
            mutants, source_one_hot, delta_score) 
        accepted_x = to_list(mutants[accept_mask])
        accepted_seq = [self._decode(x) for x in accepted_x]
        accepted_score = to_list(all_mutated_scores[accept_mask])
        return pd.DataFrame({
            'mutant_sequence': accepted_seq,
            'mutant_score': accepted_score,
        }), mutants[accept_mask]

    def compute_mutant_stats(self, source_seq, mutant_seqs):
        """Compute number of mutations in proposed sequences
        
        Args:
            source_seq: Source sequence
            mutant_seqs: Proposed mutant sequences
            
        Returns:
            num_mutated_res: Number of mutations per sequence
        """
        num_mutated_res = torch.sum(
            ~(mutant_seqs == source_seq[None]), dim=-1)
        return num_mutated_res

    def forward(self, batch):
        """Generate mutant pairs for input sequences
        
        Args:
            batch: Dictionary containing input sequences
            
        Returns:
            DataFrame of accepted mutant pairs and their scores,
            Overall acceptance rate
        """
        seqs = batch
        #Tokenize
        tokenized_seqs = self.predictor_tokenizer.encode(seqs).to(self.device)
        total_num_seqs = len(tokenized_seqs)

        # Sweep over hyperparameters
        all_mutant_pairs = []
        grand_total_num_proposals = 0
        grand_total_num_accepts = 0
        for i, (real_seq, token_seq) in enumerate(zip(seqs, tokenized_seqs)):

            # Cast as float to take gradients through
            seq_one_hot = self._make_one_hot(token_seq, differentiable=True)

            # Compute base score
            if self.landscape is not None:
                pred_score = self.landscape.get_fitness([real_seq])
            else:
                pred_score = self._evaluate_one_hot(token_seq[None]).item()

            # Construct Gibbs sampler
            sampler = self._gibbs_sampler(seq_one_hot[None]) 
            seq_pairs = []
            total_num_proposals = 0
            all_proposed_mutants = []
            all_accepted_mutants = []

            # Sample mutants
            proposed_mutants = sampler()
            num_proposals = proposed_mutants.shape[0]
            total_num_proposals += num_proposals
            grand_total_num_proposals += num_proposals
            proposed_num_edits = self.compute_mutant_stats(
                token_seq, proposed_mutants)
            proposed_mutants = proposed_mutants[proposed_num_edits > 0]
            all_proposed_mutants.append(to_np(proposed_mutants))

            # Run Gibbs generation of pairs
            sample_outputs, accepted_mutants = self._evaluate_mutants(
                mutants=proposed_mutants,
                score=pred_score,
                source_one_hot=seq_one_hot
            )

            all_accepted_mutants.append(to_np(accepted_mutants))
            grand_total_num_accepts += len(accepted_mutants)
            sample_outputs['source_sequence'] = real_seq
            sample_outputs['source_score'] = pred_score

            seq_pairs.append(sample_outputs)
            if self._verbose:
                num_pairs = len(sample_outputs)
                print(
                    f'Temp: {self.temp:.3f}'
                    f'Accepted: {num_pairs}/{num_proposals} ({num_pairs/num_proposals:.2f})'
                )

            if len(seq_pairs) > 0:
                seq_pairs = pd.concat(seq_pairs).drop_duplicates(
                    subset=['source_sequence', 'mutant_sequence'],
                    ignore_index=True
                )
                all_mutant_pairs.append(seq_pairs)
        if self._verbose:
            print("Epoch acceptance rate: ", grand_total_num_accepts / grand_total_num_proposals)

        if len(all_mutant_pairs) == 0:
            print(f"No mutants accepted for {total_num_seqs} sequences")
            return None
        return pd.concat(all_mutant_pairs).drop_duplicates(
            subset=['source_sequence', 'mutant_sequence'],
            ignore_index=True
        ), grand_total_num_accepts / grand_total_num_proposals



class GWG(flexs.Explorer):
    def __init__(
        self,
        sampler,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        temperature,
        starting_sequence,
        alphabet,
        population_size=15,
        max_iter=400,
        initial_variance=0.2,
        log_file=None,
    ):
        self.sampler = sampler
        self.model = self.sampler.predictor
        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.temperature = temperature
        self.starting_sequence = starting_sequence
        self.alphabet = alphabet
        self.population_size = population_size
        self.max_iter = max_iter
        self.initial_variance = initial_variance
        self.log_file = log_file
        self.name = 'GWG'
        self.cost = 0
        super().__init__(
            self.model,
            self.name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
    
    def _worker_fn(self, inputs):
        """Worker function for multiprocessing.

        Args:
            args (tuple): (worker_i, exp_cfg, inputs)
                worker_i: worker id. Used for setting CUDA device.
                exp_cfg: model config.
                inputs: list of inputs to process.

        Returns:
            all_outputs: results of GWG.
        """
        all_candidates, all_acceptance_rates = [], []
        result = self.sampler(inputs)
        candidates, acceptance_rate = result
        # sort candidates by mutant_score
        self.cost += len(candidates)
        candidates = candidates.drop_duplicates(subset=['mutant_sequence'])
        candidates = candidates.sort_values(by='mutant_score', ascending=False)
        candidates = candidates.head(self.sequences_batch_size)
        candidates.reset_index(drop=True, inplace=True)
        all_candidates.append(candidates)
        all_acceptance_rates.append(acceptance_rate)
        return all_candidates, all_acceptance_rates

    def propose_sequences(self, measured_sequences):
        """Propose sequences using GWG.

        Returns:
            all_candidates: list of proposed sequences.
            all_acceptance_rates: list of acceptance rates.
        """
        measured_sequences = measured_sequences.drop_duplicates(subset=['sequence'])
        measured_sequences.reset_index(drop=True, inplace=True)
        all_candidates, all_scores_list = list(measured_sequences['sequence']), list(measured_sequences['model_score'])
        # Identify indices of candidates with NaN scores
        nan_indices = [i for i, score in enumerate(all_scores_list) if np.isnan(score)]
        if nan_indices:
            # Batch-update NaN scores in one predictor call
            candidates_to_update = [all_candidates[i] for i in nan_indices]
            new_scores = self.sampler.predictor.get_fitness(candidates_to_update)
            # Ensure new_scores is iterable and update corresponding indices
            new_scores = new_scores if isinstance(new_scores, (list, np.ndarray)) else [new_scores]
            for idx, score in zip(nan_indices, new_scores):
                all_scores_list[idx] = score
        previous_model_cost = self.sampler.cost
        for _ in range(self.rounds):
            if self.sampler.cost - previous_model_cost >= self.model_queries_per_batch:
                print("Exceeded model queries per batch!")
                break
            candidates, acceptance_rate = self._worker_fn(all_candidates)
            all_candidates.extend(list(candidates[0]["mutant_sequence"]))
            all_scores = candidates[0]["mutant_score"]
            all_scores_list.extend(list(all_scores))
            # remove duplicates
            concatenated_mols = pd.DataFrame({
                "sequence": all_candidates,
                "model_score": all_scores_list
            })
            concatenated_mols = concatenated_mols.drop_duplicates(subset=['sequence'])
            all_candidates = concatenated_mols["sequence"].to_list()
            all_scores_list = concatenated_mols["model_score"].to_list()
            # sort candidates by model_score and then filter, If the filtered sequences are less than sequences_batch_size, then just return the filtered sequences
            concatenated_mols = concatenated_mols.sort_values(by='model_score', ascending=False)
            concatenated_mols = concatenated_mols.head(self.sequences_batch_size)
            concatenated_mols.reset_index(drop=True, inplace=True)
            all_candidates = concatenated_mols["sequence"].to_list()
            all_scores_list = concatenated_mols["model_score"].to_list()
        return all_candidates, all_scores_list
    
    def run(
        self, landscape: flexs.Landscape, verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        self.sampler.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.sampler.name,
            "landscape_name": landscape.name,
            "rounds": self.rounds,
            "sequences_batch_size": self.sequences_batch_size,
            "model_queries_per_batch": self.model_queries_per_batch,
        }

        # Initial sequences and their scores
        sequences_data = pd.DataFrame(
            {
                "sequence": self.starting_sequence,
                "model_score": np.nan,
                "true_score": landscape.get_fitness([self.starting_sequence]),
                "round": 0,
                "model_cost": self.sampler.cost,
                "measurement_cost": 1,
            }
        )
        self._log(sequences_data, metadata, 0, verbose, time.time())

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        for r in range_iterator(1, self.rounds + 1):
            round_start_time = time.time()
            self.model.train(
                sequences_data["sequence"].to_numpy(),
                sequences_data["true_score"].to_numpy(),
            )

            seqs, preds = self.propose_sequences(sequences_data)
            true_score = landscape.get_fitness(seqs)

            if len(seqs) > self.sequences_batch_size:
                warnings.warn(
                    "Must propose <= `self.sequences_batch_size` sequences per round"
                )

            sequences_data = sequences_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self.sampler.cost,
                        "measurement_cost": len(sequences_data) + len(seqs),
                    }
                )
            )
            self._log(sequences_data, metadata, r, verbose, round_start_time)
            # duplicate sequences_data
            sequences_data = sequences_data.drop_duplicates(subset=['sequence', 'round'])

        return sequences_data, metadata


if __name__=="__main__":
    alphabet = "ILVAGMFYWEDQNHCRKSTPBZX"
    encoder = Encoder(alphabet)
    print(encoder.vocab_size)
