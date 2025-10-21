import editdistance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pprint
import numpy as np
import json
import flexs
from flexs import baselines
import flexs.landscapes.additive_aav_packaging
import flexs.utils.sequence_utils as s_utils
import argparse
import datetime


def main(args):
    if args.landscape == 'aav':
        problem = flexs.landscapes.additive_aav_packaging.registry()['blood']
        pprint.pprint(problem)
        print(problem['params'])
        landscape = flexs.landscapes.additive_aav_packaging.AdditiveAAVPackaging(**problem['params'])
        starting_sequence = landscape.wild_type
        alphabet = s_utils.AAS
    elif args.landscape == 'rna':
        problem = flexs.landscapes.rna.registry()['L14_RNA1']
        pprint.pprint(problem)
        print(problem['params'])
        starting_sequence = problem['starts'][0]
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        alphabet = s_utils.RNAA
    elif args.landscape == 'gfp':
        landscape = flexs.landscapes.BertGFPBrightness()
        starting_sequence = landscape.gfp_wt_sequence
        alphabet = s_utils.AAS
    elif args.landscape == 'tf':
        problem = flexs.landscapes.tf_binding.registry()['SIX6_REF_R1']
        pprint.pprint(problem)
        landscape = flexs.landscapes.TFBinding(**problem['params'])
        starting_sequence = problem['starts'][0]
        alphabet = s_utils.DNAA
    elif args.landscape == 'rosetta':
        problem = flexs.landscapes.rosetta.registry()['3mx7']
        landscape = flexs.landscapes.RosettaFolding(**problem['params'])
        starting_sequence = landscape.wt_pose.sequence()
        alphabet = s_utils.AAS
    elif args.landscape == 'muformer':
        landscape = flexs.landscapes.MuformerLandscape()
        starting_sequence = landscape.starting_sequence
        seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQEGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEDIPNDERDTTMPAAMATTLRKLNTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFQADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMTERNRQIAEIGASLIKHW"
        print("-"*50)
        print(f"fitness: {landscape.get_fitness([seq])}")
        print("-"*50)
        alphabet = s_utils.AAS
    else:
        raise ValueError('Unknown landscape')
    print(f"starting_sequence length: {len(starting_sequence)}")
    def make_explorer(sequences_batch_size, model_queries_per_batch):
        if args.method == 'adalead':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.Adalead(
                    cnn,
                    rounds=10,
                    mu=1,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=sequences_batch_size,
                    model_queries_per_batch=model_queries_per_batch,
                    alphabet=alphabet,
                    log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
                )
        elif args.method == 'cmaes':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.CMAES(
                cnn,
                population_size=10,
                max_iter=200,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'dynappo':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.DynaPPO(  # DynaPPO has its own default ensemble model, so don't use CNN
                    model=cnn,
                    landscape=landscape,
                    env_batch_size=10,
                    num_model_rounds=10,
                    rounds=10,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=sequences_batch_size,
                    model_queries_per_batch=model_queries_per_batch,
                    alphabet=alphabet,
                    log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
                )
        elif args.method == 'cbas':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')

            vae = baselines.explorers.VAE(len(starting_sequence), alphabet=alphabet, epochs=10, verbose=False)
            return baselines.explorers.CbAS(
                cnn,
                vae,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'BO':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.BO(
                model=cnn,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'pex':
            model = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            pex_args = argparse.Namespace(
                num_queries_per_round=sequences_batch_size,
                num_model_queries_per_round=model_queries_per_batch,
                batch_size=sequences_batch_size,
                num_random_mutations=2,
                frontier_neighbor_size=5,
            )
            return baselines.explorers.ProximalExploration(
                args=pex_args,
                model=model,
                rounds=10,
                alphabet=alphabet,
                starting_sequence=starting_sequence,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'gwg':
            encoder = flexs.baselines.explorers.Encoder(alphabet)
            oracle_model = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                            num_filters=32, hidden_size=100, loss='MSE')
            sampler = flexs.baselines.explorers.GwgPairSampler(oracle_model, 10, sequences_batch_size=sequences_batch_size, model_queries_per_batch=model_queries_per_batch, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',)
            return flexs.baselines.explorers.GWG(sampler=sampler, rounds=10, sequences_batch_size=sequences_batch_size, model_queries_per_batch=model_queries_per_batch, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',)
        elif args.method == 'evoplay':
            evoplay_args = argparse.Namespace(
                rounds=10,
                num_queries_per_round=sequences_batch_size,
                batch_size=sequences_batch_size,
                starting_sequence=starting_sequence,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
            model = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                            num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.Evoplay(
                start_seq=starting_sequence,
                alphabet=alphabet,
                model=model,
                trust_radius=10,
                args=evoplay_args,
            )

        elif args.method == 'musearch_gt':
            musearch_args = argparse.Namespace(
                score_threshold=-np.inf,
                horizon=5,
            )
            oracle_model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.MuSearch(
                args=musearch_args,
                oracle_model=oracle_model,
                alphabet=alphabet,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                rounds=10,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'musearch':
            musearch_args = argparse.Namespace(
                score_threshold=-np.inf,
                horizon=5,
            )
            oracle_model = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.MuSearch(
                args=musearch_args,
                oracle_model=oracle_model,
                alphabet=alphabet,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                rounds=10,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'adalead_gt':
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.Adalead(
                    model,
                    rounds=10,
                    mu=1,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=sequences_batch_size,
                    model_queries_per_batch=model_queries_per_batch,
                    alphabet=alphabet,
                    log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
                )
        elif args.method == 'cmaes_gt':
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.CMAES(
                model,
                population_size=10,
                max_iter=200,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'dynappo_gt':
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.DynaPPO(  # DynaPPO has its own default ensemble model, so don't use CNN
                    model=model,
                    landscape=landscape,
                    env_batch_size=10,
                    num_model_rounds=10,
                    rounds=10,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=sequences_batch_size,
                    model_queries_per_batch=model_queries_per_batch,
                    alphabet=alphabet,
                    log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
                )
        elif args.method == 'cbas_gt':
            model = flexs.LandscapeAsModel(landscape)

            vae = baselines.explorers.VAE(len(starting_sequence), alphabet=alphabet, epochs=10, verbose=False)
            return baselines.explorers.CbAS(
                model,
                vae,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'BO_gt':
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.BO(
                model=model,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                alphabet=alphabet,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'pex_gt':
            model = flexs.LandscapeAsModel(landscape)
            pex_args = argparse.Namespace(
                num_queries_per_round=sequences_batch_size,
                num_model_queries_per_round=model_queries_per_batch,
                batch_size=sequences_batch_size,
                num_random_mutations=2,
                frontier_neighbor_size=5,
            )
            return baselines.explorers.ProximalExploration(
                args=pex_args,
                model=model,
                rounds=10,
                alphabet=alphabet,
                starting_sequence=starting_sequence,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        elif args.method == 'gwg_gt':
            model = flexs.LandscapeAsModel(landscape)
            encoder = flexs.baselines.explorers.Encoder(alphabet)
            oracle_model = model
            sampler = flexs.baselines.explorers.GwgPairSampler(oracle_model, 10, sequences_batch_size=sequences_batch_size, model_queries_per_batch=model_queries_per_batch, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file=f'efficiency/{args.method}/{args.landscape}/10_10.csv')
            return flexs.baselines.explorers.GWG(sampler=sampler, rounds=10, sequences_batch_size=sequences_batch_size, model_queries_per_batch=model_queries_per_batch, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',)
        elif args.method == 'evoplay_gt':
            evoplay_args = argparse.Namespace(
                rounds=10,
                num_queries_per_round=sequences_batch_size,
                batch_size=sequences_batch_size,
                starting_sequence=starting_sequence,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.Evoplay(
                start_seq=starting_sequence,
                alphabet=alphabet,
                model=model,
                trust_radius=10,
                args=evoplay_args,
            )
        elif args.method == 'pure_random':
            model = flexs.LandscapeAsModel(landscape)
            return baselines.explorers.PureRandom(
                alphabet=alphabet,
                model=model,
                n=5,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=sequences_batch_size,
                model_queries_per_batch=model_queries_per_batch,
                log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}_{args.run}.csv',
            )
        else:
            raise ValueError('Unknown method')
                



    results = flexs.evaluate.efficiency(landscape, make_explorer, budgets=[(args.sequences_batch_size, args.model_queries_per_batch)])
    # results = flexs.evaluate.efficiency(landscape, make_explorer, budgets=[(100, 500), (100, 5000),(1000, 5000),(1000, 10000)])
    print(results)
    # with open(f'efficiency/{args.method}/{args.landscape}.json', 'w') as f:
    #     json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='adalead')
    parser.add_argument('--landscape', type=str, default='rna')
    parser.add_argument('--sequences_batch_size', type=int, default=100)
    parser.add_argument('--model_queries_per_batch', type=int, default=5000)
    parser.add_argument('--run', type=str, default=0)
    args = parser.parse_args()
    main(args)
