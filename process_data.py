import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import urllib3
import boto3
import botocore
import logging


# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa o recurso DynamoDB com a tabela de medições
aws_access_key_id = 'myId'
aws_secret_access_key = 'myKey'
dynamodb = boto3.resource(
    'dynamodb',
    region_name='sa-east-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
table = dynamodb.Table('TabelaMedicoes')
api_key = 'caa5c4279c90488d86900701232508'


class HttpRequestError(Exception):
    def __init__(self, url, message):
        self.url = url
        self.message = message
        super().__init__(f"Falha na solicitacao HTTP para {url}. Detalhes: {message}")
                       

def lambda_handler(event, context):
    
    # Carregando os dados dos sensores e variaveis de entrada
    threshold_moisture = event['LimiteUmidadeSolo']
    threshold_temperature = event['LimiteUmidade']
    threshold_humidity = event['LimitePressao']
    threshold_pressure = event['LimiteTemperatura']
    densidade_solo = event['DensidadeSolo']
    area_solo = event['AreaPlantacao']
    profundidade_solo = event['ProfundidadeSolo']
    vazao_irrigacao = event['VazaoIrrigacao']
    qtde_irrigadores = event['QuantidadeIrrigadores']
    
    logger.info("Iniciando processamento.")
    logger.info(str(event))
    
    try:
        #Buscando medicoes no banco de dados e convertendo para dataframe
        logger.info("Buscando dados das ultimas 24h.")
        dados = get_records_between_dates()
        logger.info("Count = " + str(len(dados)))
        data = pd.DataFrame(dados, columns=['SoilMoisture', 'Pressure', 'Temperature', 'Humidity'])

        # Pre-processando os dados
        logger.info("Iniciando construcao e treinamento de modelo de rede neural.")
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Separando os dados em treinamento e teste
        train_size = int(len(data_scaled) * 0.7)
        train_data = data_scaled[0:train_size, :]

        # Definindo as janelas de tempo para previsão
        window_size = 30
        X_train, Y_train = [], []
        for i in range(window_size, len(train_data)):
            X_train.append(train_data[i-window_size:i, :])
            Y_train.append(train_data[i, 0])
        X_train, Y_train = np.array(X_train), np.array(Y_train)

        # Construindo o modelo de rede neural
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compilando o modelo
        model.compile(optimizer='Adam', loss='mse')

        # Treinar o modelo
        model.fit(X_train, Y_train, epochs=50, batch_size=32)

        # Salvar o modelo treinado
        model.save('smartAgroModel.h5')

        # Salvar o scaler após o treinamento
        joblib.dump(scaler, 'scaler.pkl')

        # Carregar o modelo e o scaler antes de fazer previsões
        model = tf.keras.models.load_model('smartAgroModel.h5')
        scaler = joblib.load('scaler.pkl')

        # Fazer previsões
        logger.info("Fazendo previsoes usando o modelo treinado.")
        train_predictions = model.predict(X_train)

        # Transformar Y_train em um array bidimensional
        Y_train = np.array(Y_train).reshape(-1, 1)

        # Inverter a escala dos valores previstos
        train_predictions_inverse = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))
        
        # Chamar a função para obter a localizacao
        localizacao = get_gps_values()
        coordenadas = localizacao.split(", ")
        latitude = float(coordenadas[0])
        longitude = float(coordenadas[1])
        logger.info(f"Localizacao definida para latitude {str(round(latitude, 2))} e longitude {str(round(longitude, 2))}")
        
        # Chama a função para obter a previsão do tempo
        logger.info("Obtendo previsoes meteorologicas.")
        informacoes_previsao = obter_previsao_tempo (latitude,longitude)
        
        # Tomar decisoes e controlar a irrigacao com base nas previsoes
        logger.info("Preparando tomada de decisao abertura da solenoide.")
        tomar_decisao(train_predictions_inverse[0], threshold_moisture, threshold_temperature, threshold_humidity, threshold_pressure,
                    densidade_solo, profundidade_solo, area_solo, vazao_irrigacao, qtde_irrigadores, informacoes_previsao)
        
        return {
                'statusCode': 200,
                'body': json.dumps("Processamento concluido com sucesso.")
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



def tomar_decisao(predicao, threshold_moisture, threshold_temperature, threshold_humidity, threshold_pressure, densidade_solo,
                  profundidade_solo, area_solo, vazao_irrigacao, irrigadores, previsao_tempo):
    umidade_solo_predita = predicao[0]
    pressao_predita = predicao[1]
    temperatura_predita = predicao[2]
    umidade_predita = predicao[3]
    
    # Obtem a previsao de chuva para as proximas 6 horas e calcula valores medios entre previsao do tempo e modelagem
    previsao_chuva = any(info["chuva"] == 1 for info in previsao_tempo)
    
    if previsao_chuva == 1:
        logger.info("Calculando impacto da chuva prevista na umidade do solo.")
        umidade_solo_predita = calcular_umidade_chuva(previsao_tempo, area_solo, umidade_solo_predita, densidade_solo, profundidade_solo)
    
    logger.info("Calculando impacto das previsoes meteorologicas nos dados obtidos na rede neural.")
    temperatura_predita = calcular_valor_previsto_medio(temperatura_predita, previsao_tempo, 'temperatura')
    umidade_predita = calcular_valor_previsto_medio(umidade_predita, previsao_tempo, 'umidade_ar')
    pressao_predita = calcular_valor_previsto_medio(pressao_predita, previsao_tempo, 'pressao')
    
    previsoes = {
        'Umidade solo': round(umidade_solo_predita,2),
        'Pressao': round(pressao_predita,2),
        'Temperatura': round(temperatura_predita,2),
        'Umidade': round(umidade_predita,2)
    }
    
    logger.info(f"Previsoes: {previsoes}")

    # Definir se a solenoide sera aberta ou nao baseado nas predicoes e nos limites
    if (umidade_solo_predita < threshold_moisture):
        ativar_solenoide(umidade_solo_predita, densidade_solo, profundidade_solo, area_solo, vazao_irrigacao, threshold_moisture,
                         irrigadores)
    elif (umidade_solo_predita < (threshold_moisture * 0.9)) and ((temperatura_predita > threshold_temperature) or \
            (umidade_predita < threshold_humidity) or (pressao_predita > threshold_pressure)) and not previsao_chuva:
        ativar_solenoide(umidade_solo_predita, densidade_solo, profundidade_solo, area_solo, vazao_irrigacao, threshold_moisture,
                         irrigadores)
    else:
        desativar_solenoide()


def ativar_solenoide(umidade_solo_inicial, densidade_solo, profundidade_solo, area_solo, vazao_irrigacao, umidade_solo_final, irrigadores):
    # Continuar irrigacao
    tempo_irrigacao = calcular_tempo_irrigacao(densidade_solo, area_solo, profundidade_solo, umidade_solo_final, umidade_solo_inicial, vazao_irrigacao, irrigadores) 
    tempo_round = round(tempo_irrigacao, 2)
    realizar_post(tempo_round)
    str_tempo = str(tempo_round)
    str_umidade = str(round(umidade_solo_inicial, 2))
    logger.info(f"Ativando solenoide por {str_tempo} segundos devido a baixa umidade do solo: {str_umidade}%")


def desativar_solenoide():
    # Parar irrigacao
    realizar_post(0)
    logger.info("Desativando solenoide")
    

def calcular_tempo_irrigacao(densidade_solo, area_solo, profundidade_solo, umidade_solo_final, umidade_solo_inicial, vazao_irrigacao, irrigadores):
    # Calcular tempo que a solenoide ficara aberta com base em formulas de vazao, densidade e umidade
    logger.info("Calculando tempo de abertura da solenoide.")
    tempo_irrigacao = (densidade_solo * area_solo * profundidade_solo * ((umidade_solo_final/100) - (umidade_solo_inicial/100))) / (1000 * vazao_irrigacao * irrigadores)
    
    return tempo_irrigacao
    

def realizar_post(tempo):
    http = urllib3.PoolManager()
    headers = {'Content-Type': 'application/json'}
    url = '170.247.125.102:5580/solenoide'
    url_envio = url + '?tempo=' + str(tempo)
    logger.info(f"Fazendo solicitacao HTTP para {url_envio}")
    try:
        response = http.request('POST', url_envio, headers=headers)
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f"Status Code {response.status}")
        logger.info(f"Resposta recebida da requisicao com status {response.status}")
    except urllib3.exceptions.HTTPError as e:
        raise HttpRequestError(url_envio, str(e))
    except Exception as e:
        raise HttpRequestError(url_envio, str(e))
        

def getAllItens():
    # Função para obter todas as medições da tabela DynamoDB
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
        
    return data


def get_records_between_dates():
    
    # Obter a data e hora atual
    data_hora_atual = datetime.now()

    # Calcular a data e hora de -1 dia
    data_hora_ontem = data_hora_atual - timedelta(days=1)

    # Formatar as datas como strings no formato esperado pelo DynamoDB
    date_format = '%Y-%m-%d %H:%M:%S'

    # Obter todas as medicoes da tabela DynamoDB
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

    for medicao in data:
        medicao['Id'] = int(medicao['Id'])
        medicao['Datetime'] = datetime.strptime(medicao['Datetime'], date_format)
        medicao['SoilMoisture'] = float(medicao['SoilMoisture'])
        medicao['Pressure'] = float(medicao['Pressure'])
        medicao['Temperature'] = float(medicao['Temperature'])
        medicao['Humidity'] = float(medicao['Humidity'])

    inicio = 0

    # Ordena as medicoes por data
    data = sorted(data, key=lambda x: x['Datetime'], reverse=False)

    for medicao in data:
        if data_hora_ontem >= medicao['Datetime']:
            inicio = medicao['Id'] + 1
            
    dados_filtro = []
    
    # Filtra as medicoes com base nos IDs encontrados
    for medicao in data:
        if (medicao['Id'] >= inicio):
            dado_filtro = {
                'SoilMoisture': medicao['SoilMoisture'],
                'Pressure': medicao['Pressure'],
                'Temperature': medicao['Temperature'],
                'Humidity': medicao['Humidity']
            }
            dados_filtro.append(dado_filtro)

    # Retornar os resultados
    return dados_filtro


def obter_previsao_tempo(latitude, longitude):
    # Construa a URL da WeatherAPI com os parâmetros necessários
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    parametros = {"key": api_key, "q": f"{latitude},{longitude}", "days": 2, "aqi": "yes"}
    
    # Crie um PoolManager para gerenciar as conexões HTTP
    http = urllib3.PoolManager()

    # Faça a requisição usando o método request da urllib3
    logger.info(f"Fazendo solicitacao HTTP para {base_url}")
    try:
        resposta = http.request('GET', base_url, fields=parametros)

        # Verifique se a requisição foi bem-sucedida (status code 200)
        if resposta.status != 200:
            raise urllib3.exceptions.HTTPError(f"Status Code {resposta.status}")
            
        # Decodifique o JSON da resposta
        dados = json.loads(resposta.data.decode('utf-8'))
        
        # Obtenha a data e hora atual
        agora = datetime.now()
        
        # Extraia a previsão para 48 horas (2 dias)
        previsao_48h = []
        for dia in dados['forecast']['forecastday']:
            previsao_48h.extend(dia['hour'])
        
        # Crie um objeto para armazenar as informações
        informacoes = []

        # Imprima a previsão para as próximas 6 horas
        for hora in previsao_48h:
            data_hora_previsao = datetime.strptime(hora['time'], "%Y-%m-%d %H:%M")
            
            # Verifica se a hora da previsão é nas próximas 6 horas
            if agora <= data_hora_previsao <= agora + timedelta(hours=6):
                informacao = {
                    "datahora": hora['time'],
                    "temperatura": hora['temp_c'],
                    "umidade_ar": hora['humidity'],
                    "pressao": hora['pressure_mb'],
                    "chuva": hora['will_it_rain'],
                    "chance_chuva": hora['chance_of_rain'],
                    "precipitacao": hora['precip_mm']
                }
                informacoes.append(informacao)
            
        logger.info(f"Resposta recebida com status {resposta.status}: {informacoes}")

        # Retorne o objeto com as informações
        return informacoes
    except urllib3.exceptions.HTTPError as e:
        raise HttpRequestError(base_url, str(e))
    except Exception as e:
        raise HttpRequestError(base_url, str(e))
        

def calcular_umidade_chuva(previsao_tempo, area_solo, umidade_solo_predita, densidade_solo, profundidade_solo):
    
    # Calcular a precipitacao prevista para as proximas horas e ajustar o valor da umidade do solo
    soma_precipitacao = 0
    for dados in previsao_tempo:
        soma_precipitacao += dados['precipitacao']
    
    massa_chuva = soma_precipitacao * area_solo
    
    massa_agua_predita = (umidade_solo_predita / 100) * densidade_solo * area_solo * profundidade_solo
    
    umidade_solo_final = (massa_agua_predita + massa_chuva) * 100 / (densidade_solo * area_solo * profundidade_solo)
    
    return umidade_solo_final


def calcular_valor_previsto_medio(valor_predito, previsao_tempo, medida):
    
    # Calcular valor medio entre a modelagem e a previsao do tempo
    soma = 0
    for dados in previsao_tempo:
        soma += dados[medida]
    
    valor_previsao = soma / len(previsao_tempo)
    
    valor_medio = (valor_predito + valor_previsao) / 2
    
    return valor_medio


def get_gps_values():
    # Obtem localizacao latitude e longitude do GPS
    url = '170.247.125.102:5580/getdata'
    logger.info(f"Fazendo solicitacao HTTP para {url}")
    http = urllib3.PoolManager()
    try:
        response = http.request('GET', url)
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f"Status Code {response.status}")
        data = json.loads(response.data.decode('utf-8'))
        logger.info(f"Resposta recebida com status {response.status}: {data}")
        localizacao = data['coordinates']
        return localizacao
    except urllib3.exceptions.HTTPError as e:
        raise HttpRequestError(url, str(e))
    except Exception as e:
        raise HttpRequestError(url, str(e))
