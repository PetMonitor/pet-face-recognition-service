from os import environ
from os.path import dirname, join
from google.oauth2 import service_account
import json

CREDENTIALS_FILE_NAME = 'credentials.json'
SCOPE = ['https://www.googleapis.com/auth/drive']

def generate_credentials():
    print('Credentials env variable is None: {}'.format(environ.get('CREDENTIALS') is None))
    if environ.get('CREDENTIALS') is None:
        serviceAccountFile = get_filename(CREDENTIALS_FILE_NAME)
        return service_account.Credentials.from_service_account_file(serviceAccountFile, scopes=SCOPE)

    serviceAccountInfo = json.loads(environ['CREDENTIALS'])
    return service_account.Credentials.from_service_account_info(serviceAccountInfo, scopes=SCOPE)


#TODO: FIX THIS https://stackoverflow.com/questions/59046883/file-json-not-found
def get_filename(filename):
    output = join('/app/', filename)
    return output