#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#define runEveryus(t) for (static uint32_t _lasttime; (uint32_t)(micros() - _lasttime) >= (t); _lasttime += (t))

// WIFI 
const char* ssid = "AP101";
const char* password = "26052006";
const char* serverName = "http://192.168.2.108:8000/classify";
bool wifiConectado = false;
String ipLocal = "-";

// Pinagem e definicoes do ECG
#define ECG_PIN 1    
#define LO_PLUS 25
#define LO_MINUS 26
#define FS 100
#define DURACAO 10
#define TOTAL_AMOSTRAS (FS * DURACAO)
unsigned long Ts = 1000000 / FS;
float ecgBuffer[TOTAL_AMOSTRAS];
int bufferIndex = 0;
bool bufferCheio = false;
#define TOTAL_MEDICOES 10
int contadorMedicoes = 0;
float somaBPM = 0;
String classesRecebidas[TOTAL_MEDICOES];
bool mostrandoResultadoFinal = false;
unsigned long tempoResultado = 0;

// Pinagem display 
#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8
#define TFT_MOSI 11
#define TFT_SCK  12

SPIClass spi = SPIClass(FSPI);
Adafruit_ST7789 tft = Adafruit_ST7789(&spi, TFT_CS, TFT_DC, TFT_RST);

// GRAFICO 
#define GRAPH_TOP    40
#define GRAPH_BOTTOM 230
#define GRAPH_LEFT   10
#define GRAPH_RIGHT  310

int lastX;
int lastY;

// Definiçao dos estados 
String classeRecebida = "-";
float bpmServidor = 0;

enum EstadoSistema {
  MEDINDO,
  MOSTRANDO_RESULTADO,
  AGUARDANDO_REINICIO
};

EstadoSistema estadoAtual = MEDINDO;
unsigned long tempoAguardar = 0;
bool telaResultadoDesenhada = false;
String classeFinalGlobal = "";
int contadorNormal = 0;
int contadorAF = 0;
int contadorPVC = 0;

// BPM 
long idx = 0;
int lastPeakIndex = -1000;
float bpmLocal = 0;
#define RR_MIN_SAMPLES 50

void detectarBPM(float signal, long sampleIndex) {
  static float prev = 0;
  static float mean = 1.5;

  mean = 0.99 * mean + 0.01 * signal;
  float threshold = mean + 0.2;

  bool pico =
    (signal > threshold) &&
    (signal > prev) &&
    (sampleIndex - lastPeakIndex > RR_MIN_SAMPLES);

  if (pico) {
    int rr = sampleIndex - lastPeakIndex;
    if (lastPeakIndex > 0) {
      bpmLocal = 60.0 * FS / rr;
    }
    lastPeakIndex = sampleIndex;
  }

  prev = signal;
}

// Função do GRID(GRAFICO) 
void drawGrid() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.println("ECG");

  tft.drawRect(GRAPH_LEFT, GRAPH_TOP,
               GRAPH_RIGHT - GRAPH_LEFT,
               GRAPH_BOTTOM - GRAPH_TOP,
               ST77XX_WHITE);

  lastX = GRAPH_LEFT;
  lastY = (GRAPH_TOP + GRAPH_BOTTOM) / 2;
}

// Função de envio 
void enviarParaServidor() {

  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(serverName);
  http.addHeader("Content-Type", "application/json");

  DynamicJsonDocument doc(30000);
  JsonArray array = doc.createNestedArray("signal");

  for (int i = 0; i < TOTAL_AMOSTRAS; i++) {
    array.add(ecgBuffer[i]);
  }

  String jsonString;
  serializeJson(doc, jsonString);

  int httpResponseCode = http.POST(jsonString);

  Serial.print("HTTP Response: ");
  Serial.println(httpResponseCode);

  if (httpResponseCode == 200) {

    String response = http.getString();

    Serial.println("==== RESPOSTA BRUTA DO SERVIDOR ====");
    Serial.println(response);
    Serial.println("====================================");

    DynamicJsonDocument respDoc(256);
    deserializeJson(respDoc, response);

    classeRecebida = respDoc["classe"].as<String>();
    bpmServidor = respDoc["bpm"];

    somaBPM += bpmServidor;
    if (contadorMedicoes >= (TOTAL_MEDICOES -1)) {
      estadoAtual = MOSTRANDO_RESULTADO;
    
    }
    classesRecebidas[contadorMedicoes] = classeRecebida;
    contadorMedicoes++;

    if (contadorMedicoes >= TOTAL_MEDICOES) {
      mostrandoResultadoFinal = true;
      tempoResultado = millis();
    }
  }

  http.end();
}

// Função de Atualização
void atualizarStatusWiFi() {

  if (WiFi.status() == WL_CONNECTED) {

    if (!wifiConectado) {
      wifiConectado = true;
      ipLocal = WiFi.localIP().toString();
    }

  } else {

    wifiConectado = false;
    ipLocal = "-";
  }

  tft.fillRect(170, 10, 150, 20, ST77XX_BLACK);
  tft.setCursor(170, 10);
  tft.setTextSize(1);

  if (wifiConectado) {
    tft.setTextColor(ST77XX_GREEN);
    tft.print("WiFi OK ");
    tft.print(ipLocal);
  } else {
    tft.setTextColor(ST77XX_RED);
    tft.print("WiFi DESCONECTADO");
  }
}

// Função de CLASSE FINAL
String calcularClasseFinal() {

  int maiorContagem = 0;
  String classeFinal = "-";

  for (int i = 0; i < TOTAL_MEDICOES; i++) {

    int contagem = 0;

    for (int j = 0; j < TOTAL_MEDICOES; j++) {
      if (classesRecebidas[i] == classesRecebidas[j]) {
        contagem++;
      }
    }

    if (contagem > maiorContagem) {
      maiorContagem = contagem;
      classeFinal = classesRecebidas[i];
    }
  }

  return classeFinal;
}

// Funcao da Correcao
void corrigirEspelhamento(uint8_t madctl) { 
  tft.startWrite(); 
  tft.writeCommand(ST77XX_MADCTL); 
  tft.spiWrite(madctl); 
  tft.endWrite(); 
  }

// SETUP 
void setup() {

  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  spi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.init(240, 320);
  tft.setRotation(1);
  corrigirEspelhamento(0x28);
  drawGrid();
  WiFi.begin(ssid, password);
  tft.setCursor(10, 220);
  tft.setTextColor(ST77XX_YELLOW);
  tft.setTextSize(2);
  tft.println("Conectando WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Conectando WiFi...");
  }

  Serial.println("WiFi conectado!");

  tft.fillRect(0, 220, 320, 20, ST77XX_BLACK);
}

// LOOP
void loop() {

  atualizarStatusWiFi();

  if (digitalRead(LO_PLUS) || digitalRead(LO_MINUS)) {
    return;
  }

  switch (estadoAtual) {
    case MEDINDO:
    {
      runEveryus(Ts) {

        float sinalRaw = analogRead(ECG_PIN) * 3.3 / 4095.0;

        ecgBuffer[bufferIndex++] = sinalRaw;

        if (bufferIndex >= TOTAL_AMOSTRAS) {
          bufferIndex = 0;
          bufferCheio = true;
        }

        idx++;
        detectarBPM(sinalRaw, idx);

        int y = GRAPH_BOTTOM - 1 - (sinalRaw / 3.3) * (GRAPH_BOTTOM - GRAPH_TOP - 2);
        int x = lastX + 1;

        if (x >= GRAPH_RIGHT) {
          drawGrid();
          x = GRAPH_LEFT;
        }

        tft.drawLine(lastX, lastY, x, y, ST77XX_GREEN);
        lastX = x;
        lastY = y;
      }

      if (bufferCheio) {
        bufferCheio = false;
        enviarParaServidor();
      }

      // Tela MEDINDO
      tft.fillRect(0, 200, 320, 40, ST77XX_BLACK);
      tft.setCursor(10, 200);
      tft.setTextColor(ST77XX_YELLOW);
      tft.setTextSize(2);
      tft.print("MEDINDO ");
      tft.print(contadorMedicoes + 1);
      tft.print("/");
      tft.println(TOTAL_MEDICOES);

      int larguraBarra = map(bufferIndex, 0, TOTAL_AMOSTRAS, 0, 300);
      tft.drawRect(10, 230, 300, 8, ST77XX_WHITE);
      tft.fillRect(10, 230, larguraBarra, 8, ST77XX_GREEN);

      break;
    }

    case MOSTRANDO_RESULTADO:
    {
      if (!telaResultadoDesenhada) {

        float mediaBPM = somaBPM / TOTAL_MEDICOES;
        classeFinalGlobal = calcularClasseFinal();

        tft.fillScreen(ST77XX_BLACK);

        tft.setCursor(60, 60);
        tft.setTextSize(3);
        tft.setTextColor(ST77XX_CYAN);
        tft.println("RESULTADO");

        tft.setTextSize(2);
        tft.setCursor(40, 120);
        tft.print("Classe: ");
        tft.println(classeFinalGlobal);

        tft.setCursor(40, 160);
        tft.print("BPM Medio: ");
        tft.println(mediaBPM);

        tempoAguardar = millis();
        telaResultadoDesenhada = true;
      }

      estadoAtual = AGUARDANDO_REINICIO;
      break;
    }

    case AGUARDANDO_REINICIO:
    {
      int segundosRestantes = 30 - ((millis() - tempoAguardar) / 1000);

      tft.fillRect(0, 200, 320, 40, ST77XX_BLACK);
      tft.setCursor(50, 210);
      tft.setTextSize(2);
      tft.setTextColor(ST77XX_YELLOW);
      tft.print("Nova em: ");
      tft.print(segundosRestantes);
      tft.print("s");

      if (millis() - tempoAguardar >= 30000) {

        contadorMedicoes = 0;
        somaBPM = 0;

        contadorNormal = 0;
        contadorAF = 0;
        contadorPVC = 0;

        telaResultadoDesenhada = false;

        drawGrid();

        estadoAtual = MEDINDO;
      }

      break;
    }
  }
}