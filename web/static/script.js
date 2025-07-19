// GPT-2 模型推理界面脚本
class ModelInterface {
  constructor() {
    this.initializeElements();
    this.bindEvents();
    this.initializeApiUrl();
    this.loadModels();
  }

  initializeElements() {
    // 获取页面元素
    this.generateButton = document.getElementById("generate-button");
    this.buttonText = document.getElementById("button-text");
    this.promptInput = document.getElementById("prompt-input");
    this.resultDiv = document.getElementById("result");
    this.maxTokensInput = document.getElementById("max-tokens-input");
    this.temperatureInput = document.getElementById("temperature-input");
    this.topKInput = document.getElementById("top-k-input");
    this.topPInput = document.getElementById("top-p-input");
    this.presencePenaltyInput = document.getElementById("presence-penalty-input");
    this.frequencyPenaltyInput = document.getElementById("frequency-penalty-input");
    this.apiKeyInput = document.getElementById("api-key-input");
    this.modelSelect = document.getElementById("model-select");
    this.apiBaseUrlInput = document.getElementById("api-base-url-input");
    this.streamCheckbox = document.getElementById("stream-checkbox");
    
    // 用于流式生成的状态
    this.isGenerating = false;
    this.eventSource = null;
  }

  bindEvents() {
    // API 地址或 Key 变更时重新加载模型
    this.apiKeyInput.addEventListener("change", () => this.loadModels());
    this.apiBaseUrlInput.addEventListener("change", () => this.loadModels());
    
    // 生成按钮点击事件
    this.generateButton.addEventListener("click", () => this.generateText());
    
    // Ctrl+Enter 快捷键生成
    this.promptInput.addEventListener("keydown", (event) => {
      if (event.ctrlKey && event.key === "Enter") {
        event.preventDefault();
        this.generateText();
      }
    });
  }

  // 自动设置API地址为当前访问的地址
  initializeApiUrl() {
    if (!this.apiBaseUrlInput.value.trim()) {
      const currentHost = window.location.host;
      const currentProtocol = window.location.protocol;
      this.apiBaseUrlInput.value = `${currentProtocol}//${currentHost}`;
    }
  }

  getApiBaseUrl() {
    let baseUrl = this.apiBaseUrlInput.value.trim();
    // 移除末尾可能存在的斜杠
    if (baseUrl.endsWith("/")) {
      baseUrl = baseUrl.slice(0, -1);
    }
    return baseUrl;
  }

  async loadModels() {
    const apiKey = this.apiKeyInput.value;
    const baseUrl = this.getApiBaseUrl();

    if (!apiKey.trim() || !baseUrl) {
      this.modelSelect.innerHTML = '<option value="">请输入 API 地址和 Key</option>';
      this.modelSelect.disabled = true;
      return;
    }

    this.modelSelect.innerHTML = "<option>正在加载模型...</option>";
    this.modelSelect.disabled = true;

    try {
      const modelsUrl = `${baseUrl}/v1/models`;
      console.log(`正在从 ${modelsUrl} 加载模型...`);

      const response = await fetch(modelsUrl, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error?.message || `HTTP 错误! 状态码: ${response.status}`
        );
      }

      const models = data.data;
      this.modelSelect.innerHTML = "";

      if (!models || models.length === 0) {
        this.modelSelect.innerHTML = '<option value="">未找到可用模型</option>';
        return;
      }

      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.id;
        this.modelSelect.appendChild(option);
      });
    } catch (error) {
      console.error("加载模型列表时出错:", error);
      this.modelSelect.innerHTML = `<option value="">加载模型失败</option>`;
      this.showError("加载模型列表失败: " + error.message);
    } finally {
      this.modelSelect.disabled = false;
    }
  }

  async generateText() {
    const promptText = this.promptInput.value;
    const maxTokens = parseInt(this.maxTokensInput.value, 10) || 150;
    const temperature = parseFloat(this.temperatureInput.value) || 0.7;
    const topK = parseInt(this.topKInput.value, 10) || 50;
    const topP = parseFloat(this.topPInput.value) || 0.9;
    const presencePenalty = parseFloat(this.presencePenaltyInput.value) || 0.0;
    const frequencyPenalty = parseFloat(this.frequencyPenaltyInput.value) || 0.0;
    const apiKey = this.apiKeyInput.value;
    const modelName = this.modelSelect.value;
    const baseUrl = this.getApiBaseUrl();
    const enableStream = this.streamCheckbox.checked;

    // 验证输入
    if (!promptText.trim()) {
      this.showError("请输入有效的提示词。");
      return;
    }
    if (!apiKey.trim() || !baseUrl) {
      this.showError("请输入有效的 API 地址和 Key。");
      return;
    }
    if (!modelName) {
      this.showError("请选择一个有效的模型。");
      return;
    }

    // 如果正在生成，先停止
    if (this.isGenerating) {
      this.stopGeneration();
      return;
    }

    // 设置加载状态
    this.setLoadingState(true);
    this.clearResult();

    const requestBody = {
      model: modelName,
      messages: [{ role: "user", content: promptText }],
      max_tokens: maxTokens,
      temperature: temperature,
      top_k: topK,
      top_p: topP,
      presence_penalty: presencePenalty,
      frequency_penalty: frequencyPenalty,
      stream: enableStream,
    };

    if (enableStream) {
      // 使用流式生成
      this.generateStream(baseUrl, apiKey, requestBody);
    } else {
      // 使用传统生成
      this.generateNonStream(baseUrl, apiKey, requestBody);
    }
  }

  async generateNonStream(baseUrl, apiKey, requestBody) {
    try {
      const completionsUrl = `${baseUrl}/v1/chat/completions`;
      console.log(`发送生成请求到 ${completionsUrl}`);

      const response = await fetch(completionsUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error?.message || `HTTP 错误! 状态码: ${response.status}`
        );
      }

      const generatedText = data.choices[0]?.message?.content;
      if (generatedText) {
        this.showResult(generatedText);
      } else {
        throw new Error("API 返回的数据格式不正确，未找到生成的文本。");
      }
    } catch (error) {
      console.error("请求错误:", error);
      this.showError("发生错误: " + error.message);
    } finally {
      this.setLoadingState(false);
    }
  }

  generateStream(baseUrl, apiKey, requestBody) {
    try {
      const completionsUrl = `${baseUrl}/v1/chat/completions`;
      console.log(`发送流式生成请求到 ${completionsUrl}`);

      // 构建POST请求的fetch调用，但使用ReadableStream处理响应
      fetch(completionsUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
          Accept: "text/event-stream",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP 错误! 状态码: ${response.status}`);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          // 读取流数据
          const readStream = () => {
            reader
              .read()
              .then(({ done, value }) => {
                if (done) {
                  console.log("流式生成完成");
                  this.setLoadingState(false);
                  return;
                }

                // 解码新数据并添加到缓冲区
                buffer += decoder.decode(value, { stream: true });

                // 处理缓冲区中的完整行
                const lines = buffer.split("\n");
                buffer = lines.pop() || ""; // 保留不完整的行

                for (const line of lines) {
                  this.processStreamLine(line.trim());
                }

                // 继续读取
                readStream();
              })
              .catch((error) => {
                console.error("流读取错误:", error);
                this.showError("流式生成错误: " + error.message);
                this.setLoadingState(false);
              });
          };

          readStream();
        })
        .catch((error) => {
          console.error("流式请求错误:", error);
          this.showError("发生错误: " + error.message);
          this.setLoadingState(false);
        });
    } catch (error) {
      console.error("流式生成初始化错误:", error);
      this.showError("发生错误: " + error.message);
      this.setLoadingState(false);
    }
  }

  processStreamLine(line) {
    if (!line || line === "data: [DONE]") {
      return;
    }

    if (line.startsWith("data: ")) {
      try {
        const jsonData = line.substring(6); // 移除 "data: " 前缀
        const chunk = JSON.parse(jsonData);

        if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) {
          const delta = chunk.choices[0].delta;
          if (delta.content) {
            // 追加新内容到结果区域
            this.appendResult(delta.content);
          }
        }
      } catch (error) {
        console.error("解析流数据错误:", error, "原始数据:", line);
      }
    }
  }

  stopGeneration() {
    this.isGenerating = false;
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.setLoadingState(false);
  }

  setLoadingState(loading) {
    this.isGenerating = loading;
    this.generateButton.disabled = loading;
    this.buttonText.innerHTML = loading 
      ? '<span class="spinner"></span>正在生成...' 
      : "生成文本";
  }

  clearResult() {
    this.resultDiv.textContent = "";
    this.resultDiv.classList.remove("error");
  }

  showResult(text) {
    this.resultDiv.textContent = text;
    this.resultDiv.classList.remove("error");
  }

  appendResult(text) {
    // 流式生成时追加内容
    if (!this.resultDiv.textContent) {
      this.resultDiv.textContent = text;
    } else {
      this.resultDiv.textContent += text;
    }
    this.resultDiv.classList.remove("error");
    
    // 自动滚动到底部
    this.resultDiv.scrollTop = this.resultDiv.scrollHeight;
  }

  showError(message) {
    this.resultDiv.textContent = message;
    this.resultDiv.classList.add("error");
  }
}

// 页面加载完成后初始化
document.addEventListener("DOMContentLoaded", () => {
  new ModelInterface();
});