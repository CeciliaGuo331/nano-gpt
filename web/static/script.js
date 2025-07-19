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
    this.apiKeyInput = document.getElementById("api-key-input");
    this.modelSelect = document.getElementById("model-select");
    this.apiBaseUrlInput = document.getElementById("api-base-url-input");
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
    const apiKey = this.apiKeyInput.value;
    const modelName = this.modelSelect.value;
    const baseUrl = this.getApiBaseUrl();

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

    // 设置加载状态
    this.setLoadingState(true);
    this.clearResult();

    try {
      const requestBody = {
        model: modelName,
        messages: [{ role: "user", content: promptText }],
        max_tokens: maxTokens,
        temperature: temperature,
      };

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

  setLoadingState(loading) {
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

  showError(message) {
    this.resultDiv.textContent = message;
    this.resultDiv.classList.add("error");
  }
}

// 页面加载完成后初始化
document.addEventListener("DOMContentLoaded", () => {
  new ModelInterface();
});