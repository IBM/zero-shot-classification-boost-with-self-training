# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

DATASET_TO_CLASS_NAME_MAPPING = \
    {'20_newsgroup':
         {'0': 'atheism', '1': 'computer graphics', '2': 'microsoft windows', '3': 'pc hardware',
          '4': 'mac hardware', '5': 'windows x', '6': 'for sale', '7': 'cars', '8': 'motorcycles', '9': 'baseball',
          '10': 'hockey', '11': 'cryptography', '12': 'electronics', '13': 'medicine', '14': 'space',
          '15': 'christianity', '16': 'guns', '17': 'middle east', '18': 'politics', '19': 'religion'},

     'ag_news':
         {'Business': 'business', 'Sci/Tech': 'science and technology', 'Sports': 'sports', 'World': 'world'},

     'dbpedia':
         {'Album': 'album', 'Animal': 'animal', 'Artist': 'artist', 'Athlete': 'athlete', 'Building': 'building',
          'Company': 'company', 'EducationalInstitution': 'educational institution', 'Film': 'film',
          'MeanOfTransportation': 'mean of transportation', 'NaturalPlace': 'natural place',
          'OfficeHolder': 'office holder', 'Plant': 'plant', 'Village': 'village', 'WrittenWork': 'written work'},

     'imdb':
         {'pos': 'good', 'neg': 'bad'}
     }
