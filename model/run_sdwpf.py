import sys
import os
sys.path.append(os.path.dirname(__file__))
from lib.sdwpf import SDWPFDataset
from lib.windff import WindFFModelManager, WindFFModelConfig

import torch
import logging
logging.basicConfig(level=logging.INFO)


def main():

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_win_sz", type=int, default=6 * 24)  # 24 hours
  parser.add_argument("--output_win_sz", type=int, default=6 * 2)  # 2 hours
  parser.add_argument("--hidden_win_sz", type=int,
                      default=(6 * 24 + 6 * 2) // 2)
  parser.add_argument("--hidden_dim", type=int, default=16)
  parser.add_argument("--adj_weight_threshold", type=float, default=0.8)

  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--early_stop", type=bool, default=True)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=1e-3)

  args = parser.parse_args()

  dataset = SDWPFDataset()

  model_manager = WindFFModelManager(WindFFModelConfig(
      feat_dim=dataset.get_feat_dim(),
      target_dim=dataset.get_target_dim(),
      hidden_dim=args.hidden_dim,
      input_win_sz=args.input_win_sz,
      output_win_sz=args.output_win_sz,
      hidden_win_sz=args.hidden_win_sz,
      dtype=torch.float64
  ))

  model_manager.train(
      [dataset[0]],
      model_manager.TrainConfig(
          adj_weight_threshold=args.adj_weight_threshold,

          epochs=args.epochs,
          early_stop=args.early_stop,
          patience=args.patience,
          lr=args.lr,
          batch_sz=args.batch_size,
          val_ratio=0.2
      )
  )

  loss_1 = model_manager.evaluate(dataset[1])
  logging.info(f"Validation loss 1: {loss_1}")

  # loss_2 = model_manager.evaluate(dataset[2])
  # logging.info(f"Validation loss 2: {loss_2}")


if __name__ == "__main__":
  main()
