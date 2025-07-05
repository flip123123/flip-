// docs/js/theme-initializer.js
(() => {
    const THEME_MODE_STORAGE_KEY = "data-md-color-scheme";
    const PRIMARY_COLOR_STORAGE_KEY = "data-md-color-primary";

    // 从 localStorage 读取值
    const currentColorScheme = localStorage.getItem(THEME_MODE_STORAGE_KEY);
    const currentPrimaryColor = localStorage.getItem(PRIMARY_COLOR_STORAGE_KEY);

    // 如果 localStorage 中有值，则立即应用到 body
    if (currentColorScheme) {
        document.body.setAttribute(THEME_MODE_STORAGE_KEY, currentColorScheme);
    } else {
        // 如果 localStorage 中没有，尝试从系统偏好设置
        const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.body.setAttribute(THEME_MODE_STORAGE_KEY, prefersDarkMode ? "dark" : "default");
    }

    if (currentPrimaryColor) {
        document.body.setAttribute(PRIMARY_COLOR_STORAGE_KEY, currentPrimaryColor);
    } else {
        // 设置一个默认值，如果 localStorage 和 body 都没值
        document.body.setAttribute(PRIMARY_COLOR_STORAGE_KEY, "blue"); // 你的默认主色
    }

    // 注意：这里只负责设置 body 的属性。
    // Material 主题的 CSS 会自动响应这些属性的变化。
    // 您的另一个 extra.js (docs/js/extra.js) 可以继续处理按钮的点击事件。
})();