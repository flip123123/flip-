// docs/js/extra.js

(() => {
    // 1. 设置颜色方案 (日间/夜间)
    var colorScheme = localStorage.getItem("data-md-color-scheme");
    // 如果 localStorage 中没有，则尝试读取系统偏好或使用默认（如 slate）
    // 注意：Material for MkDocs 可能会在加载时自动处理系统偏好。
    // 你的 JS 代码是用来“维持”用户选择的。
    if (colorScheme === null) {
        // 尝试读取系统偏好 (可选，但更好)
        const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDarkMode) {
            colorScheme = "dark"; // 或者其他你在 CSS 中定义的深色主题名
        } else {
            colorScheme = "default"; // 或者你的默认浅色主题名
        }
        localStorage.setItem("data-md-color-scheme", colorScheme);
    }
    document.body.setAttribute('data-md-color-scheme', colorScheme);

    // 2. 设置主色
    var primaryColor = localStorage.getItem("data-md-color-primary");
    if (primaryColor) {
        document.body.setAttribute('data-md-color-primary', primaryColor);
    }
    // 如果没有 localStorage 记录，你也可以在这里设置一个默认的主色
    // 例如：
    // else {
    //     document.body.setAttribute('data-md-color-primary', 'blue'); // 设置一个默认主色
    // }

    // 考虑你 CSS 中添加的 active 类，以高亮当前选中的按钮
    // 注意：这个部分需要你的 HTML 结构和上面的 JS 配合。
    // 当用户点击按钮时，JS 会在 localStorage 中设置，
    // 而你的 HTML 部分可能需要一个方法来“找到”那个按钮并添加 active 类。
    // 你的示例中没有展示如何更新 "active" 类，所以此处可以先不加，或者根据需要实现。

    // 如果你想为日间/夜间主题也添加 active 状态：
    const themeButtons = document.querySelectorAll("button[data-md-color-scheme]");
    themeButtons.forEach(btn => {
        if (btn.getAttribute("data-md-color-scheme") === colorScheme) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
        // 添加点击事件监听，当用户点击时，更新 body 的属性和 localStorage，并更新 active 类
        btn.addEventListener("click", function() {
            var newScheme = this.getAttribute("data-md-color-scheme");
            document.body.setAttribute("data-md-color-scheme", newScheme);
            localStorage.setItem("data-md-color-scheme", newScheme);

            // 更新其他按钮的 active 类
            themeButtons.forEach(otherBtn => otherBtn.classList.remove("active"));
            this.classList.add("active");
        });
    });

    // 为主色按钮添加 active 类处理
    const primaryColorButtons = document.querySelectorAll(".tx-switch button.button1");
    primaryColorButtons.forEach(btn => {
        if (btn.getAttribute("data-md-color-primary") === primaryColor) {
            btn.classList.add("active");
        }
        // 添加点击事件监听，当用户点击时，更新 body 的属性和 localStorage，并更新 active 类
        btn.addEventListener("click", function() {
            var newColor = this.getAttribute("data-md-color-primary");
            document.body.setAttribute("data-md-color-primary", newColor);
            localStorage.setItem("data-md-color-primary", newColor);

            // 更新其他按钮的 active 类
            primaryColorButtons.forEach(otherBtn => otherBtn.classList.remove("active"));
            this.classList.add("active");
        });
    });

})()