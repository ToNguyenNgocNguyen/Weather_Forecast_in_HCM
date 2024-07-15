import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd

# vscode Relative path
import sys
sys.path.append("../../")

from weatherlearn.models import Pangu_lite
from data_utils import DatasetFromFolder


parser = argparse.ArgumentParser(description="Train Pangu_lite Models")
parser.add_argument("--num_epochs", default=25, type=int, help="train epoch number")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument('--pin_memory', default=False, type=bool, help="pin memory")
parser.add_argument('--weight_decay', default=1e-3, type=float, help="weight decay")



if __name__ == "__main__":
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    NUM_WORKERS = opt.num_workers
    LEARNING_RATE = opt.lr
    PIN_MEMORY = opt.pin_memory
    WEIGHT_DECAY = opt.weight_decay


    train_set = DatasetFromFolder("data", "train")
    val_set = DatasetFromFolder("data", "valid")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    land_mask, soil_type, topography = train_set.get_constant_mask()
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0) # stacks the land mask, soil type mask, and topography mask along the channel dimension (dim=0) -> a tensor shape (3, Lat, Lon) with 3 is the number of masks, and Lat and Lon are the number of latitude and longitude points, respectively.
    lat, lon = train_set.get_lat_lon()

    pangu_lite = Pangu_lite()
    print("# parameters: ", sum(param.numel() for param in pangu_lite.parameters()))

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    if torch.cuda.is_available():
        print("CUDA, activate")
        pangu_lite.cuda()
        surface_criterion.cuda()
        upper_air_criterion.cuda()

        surface_mask = surface_mask.cuda()

    # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
    optimizer = torch.optim.Adam(pangu_lite.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    results = {'loss': [], 'surface_rmse': [], 'upper_air_rmse': [], 'surface_acc': [], 'upper_air_acc': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}

        pangu_lite.train()
        for input_surface, input_upper_air, target_surface, target_upper_air in train_bar:
            batch_size = input_surface.size(0)
            if torch.cuda.is_available():
                input_surface = input_surface.cuda()
                input_upper_air = input_upper_air.cuda()
                target_surface = target_surface.cuda()
                target_upper_air = target_upper_air.cuda()

            output_surface, output_upper_air = pangu_lite(input_surface, surface_mask, input_upper_air)

            optimizer.zero_grad()
            surface_loss = surface_criterion(output_surface, target_surface)
            upper_air_loss = upper_air_criterion(output_upper_air, target_upper_air)
            # We use the MAE loss to train the model
            # The weight of surface loss is 0.25
            # Different weight can be applied for differen fields if needed
            loss = upper_air_loss + surface_loss * 0.25
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            running_results["batch_sizes"] += batch_size

            train_bar.set_description(desc="[%d/%d] Loss: %.4f" % 
                                      (epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]))

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "surface_rmse": 0, "upper_air_rmse": 0,
                              "surface_total_cov": 0, "surface_total_output_var": 0, "surface_total_target_var": 0,
                              "upper_air_total_cov": 0, "upper_air_total_output_var": 0, "upper_air_total_target_var": 0}
            
            for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, times in val_bar:
                batch_size = val_input_surface.size(0)
                if torch.cuda.is_available():
                    val_input_surface = val_input_surface.cuda()
                    val_input_upper_air = val_input_upper_air.cuda()
                    val_target_surface = val_target_surface.cuda()
                    val_target_upper_air = val_target_upper_air.cuda()

                val_output_surface, val_output_upper_air = pangu_lite(val_input_surface, surface_mask, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)  # C Lat Lon
                val_output_upper_air = val_output_upper_air.squeeze(0)  # C Pl Lat Lon

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                valing_results["batch_sizes"] += batch_size

                surface_mse = ((val_output_surface - val_target_surface) ** 2).data.mean().cpu().item()
                upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).data.mean().cpu().item()

                valing_results["surface_rmse"] += surface_mse * batch_size
                valing_results["upper_air_rmse"] += upper_air_mse * batch_size

                # Calculate anomalies
                output_anomaly = val_output_surface - torch.mean(val_output_surface, axis=0)
                target_anomaly = val_target_surface - torch.mean(val_target_surface, axis=0)
                
                # Calculate the covariance between forecast and observation anomalies
                cov = torch.mean(output_anomaly * target_anomaly).cpu().item()
                
                # Calculate the variance of forecast and observation anomalies
                output_var= torch.mean(output_anomaly ** 2).cpu().item()
                target_var = torch.mean(target_anomaly ** 2).cpu().item()
                
                valing_results['surface_total_cov'] += cov
                valing_results['surface_total_output_var'] += output_var
                valing_results['surface_total_target_var'] += target_var

                # Calculate anomalies
                output_anomaly = val_output_upper_air - torch.mean(val_output_upper_air, axis=0)
                target_anomaly = val_target_upper_air - torch.mean(val_target_upper_air, axis=0)
                
                # Calculate the covariance between forecast and observation anomalies
                cov = torch.mean(output_anomaly * target_anomaly).cpu().item()
                
                # Calculate the variance of forecast and observation anomalies
                output_var= torch.mean(output_anomaly ** 2).cpu().item()
                target_var = torch.mean(target_anomaly ** 2).cpu().item()
                
                valing_results['upper_air_total_cov'] += cov
                valing_results['upper_air_total_output_var'] += output_var
                valing_results['upper_air_total_target_var'] += target_var

                val_bar.set_description(desc="[validating] Surface RMSE: %.4f Upper Air RMSE: %.4f Surface ACC: %.4f Upper Air ACC: %.4f" % 
                                        (np.sqrt(valing_results["surface_rmse"] / valing_results["batch_sizes"]), 
                                         np.sqrt(valing_results["upper_air_rmse"] / valing_results["batch_sizes"]),
                                         valing_results['surface_total_cov'] / np.sqrt(valing_results['surface_total_output_var'] * valing_results['surface_total_target_var']),
                                         valing_results['upper_air_total_cov'] / np.sqrt(valing_results['upper_air_total_output_var'] * valing_results['upper_air_total_target_var'])
                                         ))
        
        os.makedirs("epochs", exist_ok=True)
        torch.save(pangu_lite.state_dict(), "epochs/pangu_lite_epoch_%d.pth" % (epoch))

        results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
        results["surface_rmse"].append(np.sqrt(valing_results["surface_rmse"] / valing_results["batch_sizes"]))
        results["upper_air_rmse"].append(np.sqrt(valing_results["upper_air_rmse"] / valing_results["batch_sizes"]))
        results["surface_acc"].append(valing_results['surface_total_cov'] / np.sqrt(valing_results['surface_total_output_var'] * valing_results['surface_total_target_var']))
        results["upper_air_acc"].append(valing_results['upper_air_total_cov'] / np.sqrt(valing_results['upper_air_total_output_var'] * valing_results['upper_air_total_target_var']))


        data_frame = pd.DataFrame(
            data=results, 
            index=range(1, epoch + 1)
        )
        save_root = "train_logs"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        data_frame.to_csv(os.path.join(save_root, "logs.csv"), index_label="Epoch")
