import json
import csv
import logging
import boto3
from io import StringIO
import botocore
from datetime import datetime


# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa o recurso DynamoDB com a tabela de medições e o S3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('TabelaMedicoes')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    
    print("Iniciando processamento.")
    
    try:
        # Formatação dos dados de entrada utilizados
        date_format = '%Y-%m-%d %H:%M:%S'
        dataInicio = datetime.strptime(event['DataInicio'], date_format)
        dataFim = datetime.strptime(event['DataFim'], date_format)
        dataInicio_str = dataInicio.strftime('%Y-%m-%d').replace('-', '')
        dataFim_str = dataFim.strftime('%Y-%m-%d').replace('-', '')
        
        print(f"Intervalo de datas: {dataInicio} a {dataFim}")
        
    
        # Obtém todas as medições da tabela DynamoDB
        print("Buscando todos os registros na tabela.")
        medicoes = getAllItens()
    
        for medicao in medicoes:
            medicao['Id'] = int(medicao['Id'])
            medicao['Datetime'] = datetime.strptime(medicao['Datetime'], date_format)
            medicao['SoilMoisture'] = float(medicao['SoilMoisture'])
            medicao['Pressure'] = float(medicao['Pressure'])
            medicao['Temperature'] = float(medicao['Temperature'])
            medicao['Humidity'] = float(medicao['Humidity'])
    
        inicio = 0
        fim = 0
    
        # Ordena as medições por data
        print("Ordenando medicoes e definindo intervalo dos resultados.")
        medicoes = sorted(medicoes, key=lambda x: x['Datetime'], reverse=True)
    
        # Encontra o ID da última medição que ocorreu antes da data final
        for medicao in medicoes:
            if dataFim <= medicao['Datetime']:
                fim = medicao['Id'] - 1
    
        # Ordena as medições por data novamente (agora em ordem crescente)
        medicoes = sorted(medicoes, key=lambda x: x['Datetime'], reverse=False)
    
        for medicao in medicoes:
            if dataInicio >= medicao['Datetime']:
                inicio = medicao['Id'] + 1
    
        dadosCsv = []
    
        # Filtra as medições com base nos IDs encontrados
        for medicao in medicoes:
            if (medicao['Id'] >= inicio) and (medicao['Id'] <= fim):
                dadoCsv = {
                    'SoilMoisture': medicao['SoilMoisture'],
                    'Pressure': medicao['Pressure'],
                    'Temperature': medicao['Temperature'],
                    'Humidity': medicao['Humidity']
                }
                dadosCsv.append(dadoCsv)
                
        print(f"Dados inseridos no CSV: {json.dumps(dadosCsv)}")
    
        # Cria um objeto StringIO para escrever o CSV
        csv_string = StringIO()
        csv_writer = csv.DictWriter(csv_string, fieldnames=dadosCsv[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(dadosCsv)
    
        # Coloca o objeto CSV no bucket S3
        bucket_name = 'dados-smartagrowater-mba'
        file_key = 'dados' + dataInicio_str + dataFim_str + '.csv'
        print(f"Enviando arquivo CSV para o S3: {file_key}")
        s3.put_object(Body=csv_string.getvalue(), Bucket=bucket_name, Key=file_key)
    
        return {
            'statusCode': 200,
            'body': json.dumps(f'O arquivo CSV foi carregado para o S3: {file_key}')
        }
    except botocore.exceptions.ClientError as e:
        # Tratamento de erro específico para exceções relacionadas ao cliente AWS
        error_message = f"Erro ao interagir com a AWS: {str(e)}"
        logger.error(error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }

    except Exception as e:
        # Tratamento de erro genérico para outras exceções
        error_message = f"Erro inesperado: {str(e)}"
        logger.error(error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }

def getAllItens():
    # Função para obter todas as medições da tabela DynamoDB
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
        
    return data
