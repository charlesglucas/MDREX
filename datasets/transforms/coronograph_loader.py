import os
import numpy as np
from astropy.io import fits

class CoronographLoader:
    """
    Load and preprocess coronograph transmission maps lazily (one init per worker).
    Absolutely no heavy work in __init__.
    """
    def __init__(self, path_coronograph):
        self.path = path_coronograph
        self.h2h3 = None
        self.k1k2 = None

    def _lazy_init(self):
        """
        Called once per worker, after the fork.
        Loads FITS files into memory safely.
        """
        h2h3_path = os.path.join(
            self.path,
            "sphere_irdis_h2_h3_coronagraph_transmission_map.fits"
        )
        k1k2_path = os.path.join(
            self.path,
            "sphere_irdis_k1_k2_coronagraph_transmission_map.fits"
        )

        # Load both maps once per worker
        with fits.open(h2h3_path, memmap=False) as hdul:
            self.h2h3 = np.array(hdul[0].data, dtype=np.float32)

        with fits.open(k1k2_path, memmap=False) as hdul:
            self.k1k2 = np.array(hdul[0].data, dtype=np.float32)

    def __call__(self, item):
         # Lazy init: executed once per worker
        if self.h2h3 is None:
            self._lazy_init()

        _, _, H, W = item["frame"].shape
        lbda = item["lbda"]

        # Select appropriate coronograph file based on wavelength
        if 1.59 <= min(lbda) <= 1.68 and 1.59 <= max(lbda) <= 1.68:
            mask = self.h2h3
        elif 2.0 <= min(lbda) <= 2.3 and 2.0 <= max(lbda) <= 2.3:
            mask = self.k1k2
        else:
            raise ValueError(f"Wavelength range not supported: {lbda}")

        # Crop the mask to match frame size
        deltaH = (mask.shape[1] - H) // 2
        deltaW = (mask.shape[2] - W) // 2
        mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W]

        # Select mask slice based on wavelength
        if lbda.size == 1:
            if (1.59 <= lbda <= 1.63) or (2.0 <= lbda <= 2.15):
                mask = mask[0]
            elif (1.63 <= lbda <= 1.68) or (2.15 <= lbda <= 2.3):
                mask = mask[1]

        item["coronograph_mask"] = mask.astype(np.float32)
        # item["coronograph_mask"] = np.zeros((256,256), dtype=np.float32)

        return item
