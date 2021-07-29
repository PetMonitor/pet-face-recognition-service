import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from src.main.drive import Authenticator

    
def list_all_files():
    credentials = Authenticator.generate_credentials()
    service = build('drive', 'v3', credentials=credentials)
    fileList = service.files().list().execute()

    print('RESPONSE WAS {0}'.format(fileList))
    return fileList

def get_file_ids_for_file_extension(extension):
    fileList = list_all_files()['files']
    fileIdsForExtension = []
    for file in fileList:
        if extension in file['name']:
            fileIdsForExtension.append(file['id'])
    return fileIdsForExtension

def download_file(targetPath, fileId):
    credentials = Authenticator.generate_credentials()
    service = build('drive', 'v3', credentials=credentials)
    request = service.files().get_media(fileId=fileId)
    fh = io.FileIO(targetPath, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print('Download {}.', int(status.progress() * 100))