# Comparing Non-Bayesian Partisanship Classification Approaches

This project uses the doc2vec approach to vectorize politician's tweets and then trains these on common supervised classifiers to categorize tweets into democratic vs republican. I also used unsupervised clustering too see how these clusters fare in splitting the tweets between partisan lines. As for the supervised classifiers, the ones tested are logistic regression, k  nearest neighbors,random forests, and support vector machine.

## Getting Started

### Prerequisites

You will need to install the following packages: nltk, numpy, gensim, and sklearn.

```
sudo pip install -U nltk
sudo pip install -U numpy
pip install --upgrade gensim
pip install -U scikit-learn
```

### Tweet Processing

The tweets already processed for a small doc2vec model are provided at [train8-dem.txt](train8-dem.txt), [train8-rep.txt](train8-rep.txt), [test4-dem.txt](test4-dem.txt), and [test4-rep.txt](test4-rep.txt).
The processed tweets to train a large doc2vec model are [train825-dem.txt](train825-dem.txt), [train825-rep.txt](train825-rep.txt), [test4-dem.txt](test4-dem.txt), and [test4-rep.txt](test4-rep.txt).

If you want replicate processing data to train on the small model yourself, however, the raw datasets can be found at [dataset8.csv](dataset8.csv) and [dataset4.csv](dataset4.csv). To process the raw data, execute:

```
python csv2partisan_train8.py
python csv2partisan_test4.py
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
