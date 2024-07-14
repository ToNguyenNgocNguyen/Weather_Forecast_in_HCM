import torch


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon

    This function crops a 3D tensor to a specified resolution by taking the center of each
    tensor and taking a cube of the specified resolution from the center. The cropped tensor has shape
    (B, C, Pl_new, Lat_new, Lon_new) where Pl_new, Lat_new, and Lon_new are the resolution specified.

    The function works by calculating the padding needed for each dimension and then slicing the tensor
    to remove the padding. The padding is calculated by subtracting the resolution from the size of the
    tensor and then taking half of the remainder. The padding is then removed by slicing the tensor.
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[:, :, padding_front: Pl - padding_back, padding_top: Lat - padding_bottom,
           padding_left: Lon - padding_right]
