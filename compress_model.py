import gzip
import shutil
with open('advanced_pos_model.pth', 'rb') as f_in:
    with gzip.open('advanced_pos_model.pth.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
