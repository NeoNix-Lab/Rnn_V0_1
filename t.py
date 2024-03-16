from Models.dati import Dati as d
from Services.IchimokuDataRetriver import fetch_data_from_detailId as f, fetch_details as fd

print(fd())

data = f(14222)

obj = d(data, 'Primpo')

print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#print (obj.data_schema_txt)

#print('##@@@@@@@@@@@@@@############')
lis = d.retrive_all_from_db()
for l in lis:
    print(l.name+':  '+ f'{l.data_schema}')

