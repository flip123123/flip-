## HTML 属性

- HTML 元素可以设置**属性**
- 属性可以在元素中添加**附加信息**
- 属性一般描述于**开始标签**
- 属性总是以名称/值对的形式出现，**比如：name="value"**

比如： `<a href="http://www.runoob.com">这是一个链接</a>`  
链接的地址在 herf 属性中定义

属性的定义，单引号双引号都可以，但是当属性值中本身有双引号时，你必须得使用单引号

[HTML 属性表](https://www.runoob.com/html/html-attributes.html)直接上链接，有些可以等需要了再记

## HTML 标题

浏览器会在标题的前后自动加空行  
`<hr>` 标签在 HTML 页面中添加空白行，用于分割内容

## HTML 注释

注释写法为  
`<!--这是一个注释-->`  
其中左括号和感叹号是不可少的，但最好也把破折号和结束括号给加上

## HTML 提示

可以直接在网页任意处右键，点击查看源代码，就可以看到这个网页的实现方式

## HTML 折行

直接在段落内容中加入 `<br>` ，可以直接将内容分行

## HTML 输出

HTML 的输出页面无法通过在代码中加空格或空行来改变，所有连续的空格或空行都会被算作一个空格

## HTML 文本格式化

粗体是使用 `<b></b>` ,斜体是使用 `<i></i>`  
同样有很多文本格式化的标签，依旧[上链接](https://www.runoob.com/html/html-formatting.html)

## HTML 链接

### 基本语法、

`<a href="URL">链接文本</a>`

示例：  
`<a href="/index.html">本文本</a> 是一个指向本网站中的一个页面的链接。</p>`  
`<p><a href="https://www.microsoft.com/">本文本</a> 是一个指向万维网上的页面的链接。</p>`

最后呈现的结果是，本文本三个字是可点击跳转的链接，后面跟随着链接文本

### 链接属性

示例：  
`<a href="https://www.example.com" target="_blank" rel="noopener">新窗口打开 Example</a>`

1. href：定义链接目标
2. target：定义链接的打开方式
	1. - `_blank`: 在新窗口或新标签页中打开链接
	2. `_self`: 在当前窗口或标签页中打开链接（默认）
	3. `_parent`: 在父框架中打开链接
	4. `_top`: 在整个窗口中打开链接，取消任何框架
3. rel：定义链接与目标页面的关系
	1. `nonfollow` : 表示搜索引擎不跟踪该链接，常用于外部链接
	2. `noopener`: 防止新的浏览上下文（页面）访问`window.opener`属性和`open`方法
	3. `noreferrer`: 不发送referer header（即不告诉目标网站你从哪里来的）
	4. `noopener noreferrer`: 同时使用`noopener`和`noreferrer`  
剩下的内容也[上链接](https://www.runoob.com/html/html-links.html)吧