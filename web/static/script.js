// 确保在DOM加载完成后再执行脚本
document.addEventListener('DOMContentLoaded', () => {

    // 获取页面上的主要元素
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt-input');
    const maxLengthInput = document.getElementById('max-length-input');
    const responseText = document.getElementById('response-text');
    const loader = document.getElementById('loader');

    // 为“生成文本”按钮添加点击事件监听器
    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        const maxLength = parseInt(maxLengthInput.value, 10);

        // 简单的输入验证
        if (!prompt) {
            alert('请输入提示文本！');
            return;
        }

        // 开始生成，显示加载动画并禁用按钮
        loader.style.display = 'block';
        responseText.innerText = '';
        generateBtn.disabled = true;
        generateBtn.innerText = '正在生成中...';

        try {
            // 使用fetch API调用后端的 /generate 端点
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // 将用户的输入打包成JSON格式发送
                body: JSON.stringify({
                    prompt: prompt,
                    max_length: maxLength
                }),
            });

            // 检查API的响应状态
            if (!response.ok) {
                // 如果API返回错误，则抛出异常
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP 错误! 状态码: ${response.status}`);
            }

            // 解析API返回的JSON数据
            const data = await response.json();
            
            // 将模型生成的文本显示在页面上
            responseText.innerText = data.generated_text;

        } catch (error) {
            // 如果发生任何错误，在页面上显示错误信息
            console.error('生成失败:', error);
            responseText.innerText = `生成失败，请检查后端服务是否正常。\n\n错误详情: ${error.message}`;
        } finally {
            // 无论成功还是失败，都隐藏加载动画并恢复按钮
            loader.style.display = 'none';
            generateBtn.disabled = false;
            generateBtn.innerText = '生成文本';
        }
    });
});