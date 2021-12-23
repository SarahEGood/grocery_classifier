# Grocery Item Classifier

The image classifier being created within this project aims to sort images of "loose" grocery items (i.e. individual fruits, vegetables, bakery items, etc.) by assigning them general to specific labels (i.e. fruit, apple, royal gala).

The dataset is hosted on [figshare](https://figshare.com/projects/Grocery_Classifier_Data/128429). Code for the model is stored in the .ipynb notebook here, along with the source for the web app made with Flask as posted on [Google Cloud Platform](http://grocery-classifier-335420.uc.r.appspot.com/).

## Image Dataset
The Image dataset can be found [here](https://figshare.com/projects/Grocery_Classifier_Data/128429).

## Installation and Usage

### Online Version

A tentative public version of this app is available on [Google Cloud Platform](http://grocery-classifier-335420.uc.r.appspot.com/). Note that this version may contain bugs.

### Offline Version
To run offline, clone repo and install requirements.

```
git clone https://github.com/SarahEGood/grocery_classifier.git
pip install -r requirements.txt
```

From here, the app can be run locally at the root directory with Flask like so:

```
flask run
```


## License
See LICENSE.txt