import torch
import argparse


def get_configs(parser=argparse.ArgumentParser()):
    parser.add_argument(
        "--workers", type=int, default=0, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size, default=16"
    )

    parser.add_argument(
        "--epoch", type=int, default=300, help="number of epochs, default=100"
    )

    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate, default=5e-4"
    )

    parser.add_argument(
        "--patience", type=int, default=10, help="the early stop patience"
    )

    parser.add_argument(
        "--step_size", type=int, default=10, help="the learning rate decay step size"
    )

    parser.add_argument(
        "--gamma", type=float, default=0.9, help="the gamma value of scheduler"
    )
    
    parser.add_argument("--alpha", type=float, default=0.1, help="the rate of fussion")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/processed data/all_od_3.pkl",
        help="the data path",
    )

    parser.add_argument(
        "--se_path",
        type=str,
        default="../data/processed data/zone_embedding.npy",
        help="the static embedding path",
    )

    parser.add_argument(
        "--pe_path",
        type=str,
        default="../data/processed data/poi_f.csv",
        help="poi embedding path",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="../model",
        help="the model path",
    )

    parser.add_argument("--model_name", type=str, default="vanilla", help="model name")

    parser.add_argument(
        "--loss_path", type=str, default="../loss", help="the loss file path"
    )

    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="device",
    )
    
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=66,
        help="The node num",
    )
    
    parser.add_argument(
        "--in_seq_len",
        type=int,
        default=24,
        help="The sequence length of transformers input",
    )
    parser.add_argument(
        "--out_seq_len",
        type=int,
        default=24,
        help="The sequence length of transformers output",
    )

    parser.add_argument(
        "--slide_len",
        type=int,
        default=24,
        help="the slide window length of data sample",
    )

    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=1,
        help="number of transformer encoder layers",
    )
    
    parser.add_argument(
        "--n_feature",
        type=int,
        default=132,
        help="The num of x feature",
    )
    
    parser.add_argument(
        "--s_feature",
        type=int,
        default=128,
        help="The num of se feature",
    )
    
    parser.add_argument(
        "--p_feature",
        type=int,
        default=13,
        help="The num of poi feature",
    )
    
    parser.add_argument(
        "--t_feature",
        type=int,
        default=31,
        help="The num of te feature",
    )
    
    parser.add_argument(
        "--d_model", type=int, default=128, help="dimention of output layer"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="dimention of hidden channels"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="number of attention heads"
    )
    parser.add_argument("--dropout", type=int, default=0.1, help="value of dropout")

    parser.add_argument(
        "--test_size", type=float, default=0.235, help="the train and test split size"
    )
    
    parser.add_argument("--his_window", type=int, default=1, help="this history window of cross day")
    
    parser.add_argument("--attention_window", type=int, default=7, help="this attention window size")

    cfg = parser.parse_known_args()[0]

    return cfg
