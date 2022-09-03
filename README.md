# pet-face-recognition-service

## Basic Set Up

This project attempts to download a pre-existing model to generate face embeddings. In order to do that
it requires that the user provides credentials to access the Google Drive API.

Generate the development app and a secret key with permissions to read and download documents using
the Google Drive API. Once the key has been generated, you can provide it on startup via evironment variable
by storing key to the variable CREDENTIALS. Alternatively, save the provided file under the name 'credentials.json' and place it in the project's root directory.

## Requirements

- Docker and Docker Compose

## Run Project Locally

- Open a terminal and navigate to the project's root directory. 
- Run `docker-compose up`

