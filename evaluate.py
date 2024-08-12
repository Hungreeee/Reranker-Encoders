import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from encoders import BiEncoder, CrossEncoder, PolyEncoder
from transformers import AutoModel, AutoTokenizer

from utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder", required=True, type=str, help="The encoder model to be trained")
    parser.add_argument("--poly_m", required=True, type=str, help="Poly encoder's number of global features")

    parser.add_argument("--base_transformer", default="prajjwal1/bert-tiny", type=str, help="The base BERT transformer.")
    parser.add_argument("--dataset", default="ms_marco", type=str)
    parser.add_argument("--dataset_configs", default="./dataset_configs.cfg", type=str)
    parser.add_argument("--output_dir", default="./model_checkpoints/", type=str)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--num_epoch", default=15, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=100, type=int)

    parser.add_argument("--log_results", default=True, type=bool, help="Whether to log the validation results during training.")
    parser.add_argument('--device', type=int, default="cpu")

    args = parser.parse_args()

    base_model = AutoModel(args.base_transformer)
    base_tokenizer = AutoModel(args.base_transformer)

    dataset_configs = read_config_file(args.dataset_configs)
    trainset, valset = load_huggingface_dataset(args.dataset, dataset_configs, base_tokenizer, args["max_query_length"])

    trainloader = DataLoader(trainset, batch_size=args["batch_size"], shuffle=True)
    valloader = DataLoader(trainset, batch_size=args["batch_size"], shuffle=True)

    if args.encoder == "bi":
        model = BiEncoder(transformer=base_model)
    elif args.encoder == "cross":
        model = CrossEncoder(transformer=base_model)
    elif args.encoder == "poly":
        model = PolyEncoder(transformer=base_model, num_global_features=args.poly_m)

    model_parameters = model.parameters()
    print(f"Number of trainable parameters: {count_parameters(model_parameters)}")

    optimizer = optim.Adam(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training started...")
    train_loss, val_loss = train(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epoch=args["num_epoch"],
        device=args["device"],
        eval=True
    )

    print("Model saved to...")

