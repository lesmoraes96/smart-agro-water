import json
import boto3
import botocore
import urllib3
from datetime import datetime, timezone, timedelta
import logging
from decimal import Decimal, Inexact, ROUND_HALF_UP


# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar o recurso DynamoDB com a tabela de medições
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('TabelaMedicoes')

class HttpRequestError(Exception):
    def __init__(self, url, message):
        self.url = url
        self.message = message
        super().__init__(f"Falha na solicitacao HTTP para {url}. Detalhes: {message}")

def lambda_handler(event, context):
    
    print("Iniciando processamento.")
    
    try:
        # Formatação dos dados de entrada utilizados
        url = '170.247.125.102:5580/getdata'
        
        # Obter a data e hora atual em UTC
        execucao_utc = datetime.now(timezone.utc)
        
        # Configurar o deslocamento de tempo para Brasília (UTC-3)
        fuso_horario_brasilia = timezone(timedelta(hours=-3))
        
        # Converter a data e hora para o fuso horário de Brasília
        execucao_brasilia = execucao_utc.astimezone(fuso_horario_brasilia)
        
        # Formatar a data e hora como uma string
        execucao = execucao_brasilia.strftime('%Y-%m-%d %H:%M:%S')
         
        #Conectar ao IP externo para obter medicoes
        print(f"Fazendo solicitacao HTTP para {url}")
        medicoes = get_sensor_values(url)
        
        # Inserir medições no DynamoDB
        print("Encontrando ultimo ID no banco de dados.")
        last_id = find_max_numeric_id()
        print("Salvando medicoes no banco de dados.")
        put_item_dynamodb(medicoes, last_id, execucao)
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Medicoes importadas com sucesso. Id = {last_id}')
        }
    except botocore.exceptions.ClientError as e:
         # Tratamento de erro específico para exceções relacionadas ao cliente AWS
        error_message = f"Erro ao interagir com a AWS: {str(e)}"
        logger.error(error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }
        
    except HttpRequestError as e:
        # Tratamento de erro especifico para falhas nas solicitações HTTP
        error_message = f"Erro na solicitacao HTTP para {e.url}. Detalhes: {e.message}"
        logger.error(error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }
    except Exception as e:
        # Tratamento de erro generico para outras excecoes
        error_message = f"Erro inesperado: {str(e)}"
        logger.error(error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }


def put_item_dynamodb(medicoes, last_id, execucao):
    # Insere medicoes no DynamoDB
    dadoDb = {
        'Id': int(last_id),
        'Datetime': execucao,
        'SoilMoisture': Decimal(str(medicoes['soilMoisture'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'Pressure': Decimal(str(medicoes['pressure'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'Temperature': Decimal(str(medicoes['temperature'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'Humidity': Decimal(str(medicoes['humidity'])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    }
    table.put_item(Item=dadoDb)
    print(f"Dado inserido no DynamoDB: {dadoDb}")
        

def find_max_numeric_id():
    # Inicializa o valor maximo do ID
    max_numeric_id = 0

    # Executa uma operacao de scan para percorrer todos os itens na tabela
    response = table.scan()

    # Verifica cada item na resposta
    for item in response['Items']:
        # Converte o valor do ID para inteiro
        numeric_id = int(item.get('Id', 0))

        # Atualiza o valor máximo do ID, se necessario
        if numeric_id > max_numeric_id:
            max_numeric_id = numeric_id

    # Se houver mais itens para escanear, continua a operação de scan
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])

        # Verifica cada item na resposta
        for item in response['Items']:
            # Converte o valor do ID para inteiro
            numeric_id = int(item.get('Id', 0))

            # Atualiza o valor máximo do ID, se necessário
            if numeric_id > max_numeric_id:
                max_numeric_id = numeric_id

    max_numeric_id = max_numeric_id + 1
    
    return max_numeric_id
    
    
def get_sensor_values(url):
    # Obtem medicoes dos sensores
    http = urllib3.PoolManager()
    try:
        response = http.request('GET', url)
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f"Status Code {response.status}")
        medicoes = json.loads(response.data.decode('utf-8'))
        print(f"Resposta recebida com status {response.status}: {medicoes}")
        return medicoes
    except urllib3.exceptions.HTTPError as e:
        raise HttpRequestError(url, str(e))
    except Exception as e:
        raise HttpRequestError(url, str(e))
