# General

Built up from fastTgcnBase. Runs an abbreviated version of the model on a smaller set of data. Used for initial proof of concept.

Run via:

```shell
python train.py
```

## Changes from fastTgcnBase

* The code is set up to run on a machine with 3 GPUs. My machine only has one. In the train.py file there is a statement os.environ["CUDA_VISIBLE_DEVICES"] = '2' which tells the file to operate on the 3rd indexed GPU. Similarly there is a statement in the dataloader telling the file to operate on the 2nd GPU. I have commented these out so that the files default to working on the single GPU that I have. This could also be manually overwritten by changing the 2 and the 1 in the statements to 0.  
* The Dataloader() function is set up to load in the data via parallel processing. This is invoked by the num_workers argument. This did not work on my machine, likely due to the fact that the parallelism was set up for a more high powered machine. I have changed this argument to 0 for the moment. 
* For now I am working with only the lower data and have set the filepaths to reflect that 
* On my machine, the full training and testing data takes a long time, I have decreased it to 5 training files and 2 test tiles so that we can look at some output. It does not have to be good output, just wanting to see what happens 
* another choke point computationally is the number of epochs. Perhaps this is no problem on a more powerful machine but on mine 301 epochs takes a very long time. I have decreased it to 31 so that we can get this running. 

# Notes

On my machine: Success with the current set up of 5 training, 2 test, and 31 epochs, the model successfully runs. 

 
On HPC: Success with the current set up of 5 training, 2 test, and 31 epochs, the model successfully runs. 



