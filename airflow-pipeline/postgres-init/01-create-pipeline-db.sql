-- Cria o usuário e o banco de dados da aplicação, executado na inicialização do container Postgres
-- Valores devem coincidir com APP_DB_USER, APP_DB_PASSWORD e APP_DB_NAME em .env

-- Cria usuário da aplicação
CREATE USER app_user WITH ENCRYPTED PASSWORD 'senha_super_segura_para_app';

-- Cria o database da aplicação com o owner definido
CREATE DATABASE app_data OWNER app_user;