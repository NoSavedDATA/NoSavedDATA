import pandas as pd

class CSV_Writer:
    def __init__(self, write_every=10000):
        self.first_write = True
        self.write_every = write_every
    
    def write(self, running_df, file_path, force_write=False):
        # writes every 1000 and resets the df on RAM

        key = list(running_df.keys())[0]
        
        if len(running_df[key]) > self.write_every or (force_write and len(running_df[key])>0):

            df = pd.DataFrame(running_df)

            # If file exists, append to it; otherwise, create from scratch
            if self.first_write:
                df.to_csv(file_path, index=False, sep='|', mode='w')
                self.first_write=False
            else:
                df.to_csv(file_path, index=False, sep='|', mode='a', header=False)

            for key in running_df.keys():
                running_df[key].clear()
        
        return running_df
