#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
from collections import namedtuple
import requests
PARAMS = {'Authorization' : 'p0etoa9eboj8dnd5lfv56rju8a'}
sad = "https://api.sentiocloud.net/v2/"
r = requests.get(url=sad+'Matches', headers = PARAMS)
r = requests.utils.get_unicode_from_response(r)
mat = json.loads(r)[0]['id']

new_r = requests.get(url=sad+'CurrData/'+str(mat)+'/Players', headers = PARAMS)
new_r = requests.utils.get_unicode_from_response(new_r)
print new_r

"""
with open("deneme.json") as json_file:
    data=json.load(json_file)
print data[0]['referee']
"""