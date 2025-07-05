javascript
// docs/js/extra.js

(() => {
    // --- 配置 ---
    // Material for MkDocs 使用的 localStorage 键名
    const THEME_MODE_STORAGE_KEY = "data-md-color-scheme";
    const PRIMARY_COLOR_STORAGE_KEY = "data-md-color-primary";

    // 默认值
    const DEFAULT_THEME_MODE = "default"; // Material for MkDocs 默认是 "default" 或你的自定义主题名
    const DEFAULT_PRIMARY_COLOR = "blue"; // Material for MkDocs 的一个内置主色，你可以根据需要修改

    // --- 函数：获取当前页面设置的属性值 ---
    function getCurrentAttribute(attributeName) {
        return document.body.getAttribute(attributeName);
    }

    // --- 函数：设置页面属性并更新 localStorage ---
    function setPageAttribute(attributeName, value) {
        document.body.setAttribute(attributeName, value);
        localStorage.setItem(attributeName, value);
    }

    // --- 函数：更新切换按钮的 active 状态 ---
    function updateButtonActiveState(buttons, activeValue, attributeName) {
        buttons.forEach(btn => {
            const btnValue = btn.getAttribute(attributeName);
            if (btnValue === activeValue) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }

    // --- 初始化 ---
    // 1. 设置颜色方案 (日间/夜间)
    let currentColorScheme = getCurrentAttribute(THEME_MODE_STORAGE_KEY);

    if (!currentColorScheme) {
        // 如果 body 上没有，尝试从 localStorage 读取
        currentColorScheme = localStorage.getItem(THEME_MODE_STORAGE_KEY);
        if (!currentColorScheme) {
            // 如果 localStorage 中也没有，则读取系统偏好，并设置默认值
            const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
            currentColorScheme = prefersDarkMode ? "dark" : DEFAULT_THEME_MODE;
            // 注意：Material for MkDocs 可能会自己处理系统偏好，这里是为了确保你的 JS 也同步。
            // 如果 Material 主题已经设置了 body 的属性，这里获取到的就会是 Material 的值。
            // 为保险起见，我们先确保 body 有个属性
            document.body.setAttribute(THEME_MODE_STORAGE_KEY, currentColorScheme);
            localStorage.setItem(THEME_MODE_STORAGE_KEY, currentColorScheme);
        } else {
            // 从 localStorage 读取到了值，但 body 上没有，就设置给 body
            document.body.setAttribute(THEME_MODE_STORAGE_KEY, currentColorScheme);
        }
    } else {
        // 如果 body 上有，确保 localStorage 和它同步
        localStorage.setItem(THEME_MODE_STORAGE_KEY, currentColorScheme);
    }

    // 2. 设置主色
    let currentPrimaryColor = getCurrentAttribute(PRIMARY_COLOR_STORAGE_KEY);
    if (!currentPrimaryColor) {
        currentPrimaryColor = localStorage.getItem(PRIMARY_COLOR_STORAGE_KEY);
        if (!currentPrimaryColor) {
            // 如果 localStorage 和 body 上都没有，就设置一个默认值
            currentPrimaryColor = DEFAULT_PRIMARY_COLOR;
            document.body.setAttribute(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
            localStorage.setItem(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
        } else {
            // 从 localStorage 读取到了值，但 body 上没有，就设置给 body
            document.body.setAttribute(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
        }
    } else {
        // 如果 body 上有，确保 localStorage 和它同步
        localStorage.setItem(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
    }


    // --- 为主题切换按钮添加事件监听 ---
    // 查找所有用于切换颜色方案的按钮 (通常是 Material for MkDocs 的配置按钮)
    const themeModeButtons = document.querySelectorAll("button[data-md-color-scheme]");
    if (themeModeButtons.length > 0) {
        updateButtonActiveState(themeModeButtons, currentColorScheme, THEME_MODE_STORAGE_KEY);
        themeModeButtons.forEach(btn => {
            btn.addEventListener("click", function() {
                const newScheme = this.getAttribute(THEME_MODE_STORAGE_KEY);
                setPageAttribute(THEME_MODE_STORAGE_KEY, newScheme);
                updateButtonActiveState(themeModeButtons, newScheme, THEME_MODE_STORAGE_KEY);
            });
        });
    }

    // --- 为主色切换按钮添加事件监听 ---
    // 查找你的自定义主色选择按钮
    // 注意：这里假设你的主色按钮的类名是 button1 并且有 data-md-color-primary 属性
    // 如果 Material for MkDocs 主题本身有内置的颜色切换按钮，你需要找到它们的类名并适配
    // 你的 CSS 中使用了 `.tx-switch button.button1`，所以我们以此为基础
    const primaryColorButtons = document.querySelectorAll(".tx-switch button.button1");
    if (primaryColorButtons.length > 0) {
        updateButtonActiveState(primaryColorButtons, currentPrimaryColor, PRIMARY_COLOR_STORAGE_KEY);
        primaryColorButtons.forEach(btn => {
            btn.addEventListener("click", function() {
                const newColor = this.getAttribute(PRIMARY_COLOR_STORAGE_KEY);
                setPageAttribute(PRIMARY_COLOR_STORAGE_KEY, newColor);
                updateButtonActiveState(primaryColorButtons, newColor, PRIMARY_COLOR_STORAGE_KEY);
            });
        });
    }

})(); // 立即执行函数结束