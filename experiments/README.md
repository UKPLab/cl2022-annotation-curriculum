# Annotation Curricula

> This repository contains experimental software and is published for the sole purpose of giving additional background 
> details on the respective publication.

## Setup

    conda create -n ac
    source activate ac 
    conda install pip
    pip install -r requirements.txt

## Data

### Download data

You can download and preprocess the data for `SigIE` and `SPEC` by running

    make prepare

The raw datasets are available [here](http://pages.cs.wisc.edu/~bsettles/data/).
Muc7T needs to be purchased from the Linguistic Data Consortium, extracted and
put into the `data` folder. 

## Splits

The splits can be find attached. For SPEC, we use the first column as the id
(PubMed abstract id + sentence number), for SigIE the first colum as well
(the source Sig+Reply file ID) and for Muc7T the Muc7 file name + document id +
sentence id seperated by colons.

## Experiments
    



    
## Development

Run
    
    make format
    
before pushing.

