from os import environ
from os.path import dirname, join
from google.oauth2 import service_account
import json

CREDENTIALS_FILE_NAME = 'credentials.json'
SCOPE = ['https://www.googleapis.com/auth/drive']

def generate_credentials():
    if not environ.get('CREDENTIALS'):
        service_account_file = get_filename(CREDENTIALS_FILE_NAME)
        return service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPE)
    service_account_info = json.loads(environ.get('CREDENTIALS'))
    return service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPE)


#TODO: FIX THIS https://stackoverflow.com/questions/59046883/file-json-not-found
def get_filename(filename):
    here = dirname(__file__)
    output = join(here, filename)
    return output