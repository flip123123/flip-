// docs/js/extra.js

(() => {
    // --- 配置 ---
    // Material for MkDocs 使用的 localStorage 键名
    const THEME_MODE_STORAGE_KEY = "data-md-color-scheme";
    const PRIMARY_COLOR_STORAGE_KEY = "data-md-color-primary";

    // 默认值（当 localStorage 中没有时使用）
    const DEFAULT_THEME_MODE = "default"; // Material for MkDocs 默认是 "default" 或你的自定义主题名
    const DEFAULT_PRIMARY_COLOR = "blue"; // Material for MkDocs 的一个内置主色

    // --- 函数：设置页面属性并更新 localStorage ---
    function setPageAttributeAndStore(attributeName, value) {
        document.body.setAttribute(attributeName, value);
        localStorage.setItem(attributeName, value);
    }

    // --- 初始化 ---
    // 1. 设置颜色方案 (日间/夜间)
    let currentColorScheme = localStorage.getItem(THEME_MODE_STORAGE_KEY) ||
                           (window.matchMedia('(prefers-color-scheme: dark)').matches ? "dark" : DEFAULT_THEME_MODE);
    document.body.setAttribute(THEME_MODE_STORAGE_KEY, currentColorScheme);
    localStorage.setItem(THEME_MODE_STORAGE_KEY, currentColorScheme); // 确保 localStorage 有值

    // 2. 设置主色
    let currentPrimaryColor = localStorage.getItem(PRIMARY_COLOR_STORAGE_KEY) || DEFAULT_PRIMARY_COLOR;
    document.body.setAttribute(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
    localStorage.setItem(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor); // 确保 localStorage 有值


    // --- 为主题切换按钮添加事件监听 ---
    const themeModeButtons = document.querySelectorAll("button[data-md-color-scheme]");
    themeModeButtons.forEach(btn => {
        btn.addEventListener("click", function() {
            const newScheme = this.getAttribute(THEME_MODE_STORAGE_KEY);
            setPageAttributeAndStore(THEME_MODE_STORAGE_KEY, newScheme);

            // 可选：更新按钮的 active 状态 (如果你的 HTML 结构支持)
            themeModeButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // --- 为主色切换按钮添加事件监听 ---
    // 假设你的主色按钮的类名是 button1 并且有 data-md-color-primary 属性
    const primaryColorButtons = document.querySelectorAll(".tx-switch button.button1");
    primaryColorButtons.forEach(btn => {
        btn.addEventListener("click", function() {
            const newColor = this.getAttribute(PRIMARY_COLOR_STORAGE_KEY);
            setPageAttributeAndStore(PRIMARY_COLOR_STORAGE_KEY, newColor);

            // 可选：更新按钮的 active 状态 (如果你的 HTML 结构支持)
            primaryColorButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });

})();

window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
  
  document$.subscribe(() => { 
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })