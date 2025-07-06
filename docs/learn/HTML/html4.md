## HTML \<head\> 元素

\<head\> 元素包含了所有的头部标签元素。在 \<head\>元素中你可以插入脚本（scripts）, 样式文件（CSS），及各种meta信息。

可以添加在头部区域的元素标签为: \<title\>, \<style\>, \<meta\>, \<link\>, \<script\>, \<noscript\> 和 \<base\>。


## HTML \<title\> 元素

\<title\> 标签定义了不同文档的标题。

\<title\> 在 HTML/XHTML 文档中是必需的。

\<title\> 元素:

- 定义了浏览器工具栏的标题
- 当网页添加到收藏夹时，显示在收藏夹中的标题
- 显示在搜索引擎结果页面的标题


## HTML \<base\> 元素

\<base\> 标签描述了基本的链接地址/链接目标，该标签作为HTML文档中所有的链接标签的默认链接:

`<head>`
`<base href="http://www.runoob.com/images/" target="_blank">`
`</head>`


## HTML \<link\> 元素

\<link\> 标签定义了文档与外部资源之间的关系。

\<link\> 标签通常用于链接到样式表:

`<head> <link rel="stylesheet" type="text/css" href="mystyle.css"> </head>`

其中：

- **`rel="stylesheet"`** : 指明这是一个用来设置页面样式的样式表文件。
- **`type="text/css"`** : 指明这个样式表文件的内容类型是CSS（文本格式）。
- **`href="mystyle.css"`** : 指明了样式表文件的具体位置。


## HTML \<style\> 元素

\<style\> 标签定义了HTML文档的样式文件引用地址.

在\<style\> 元素中你也可以直接添加样式来渲染 HTML 文档


## HTML \<meta\> 元素

meta标签描述了一些基本的元数据。

\<meta\> 标签提供了元数据.元数据也不显示在页面上，但会被浏览器解析。

META 元素通常用于指定网页的描述，关键词，文件的最后修改时间，作者，和其他元数据。

元数据可以使用于浏览器（如何显示内容或重新加载页面），搜索引擎（关键词），或其他Web服务。

\<meta\> 一般放置于 \<head\> 区域

### \<meta\> 标签- 使用实例

为搜索引擎定义关键词:

```html
<meta name="keywords" content="HTML, CSS, XML, XHTML, JavaScript">
```

为网页定义描述内容:

```html
<meta name="description" content="教程">
```

定义网页作者:

```html
<meta name="flip" content="html">
```

每30秒钟刷新当前页面:

```html
<meta http-equiv="refresh" content="30">
```


## HTML \<script\> 元素

\<script\>标签用于加载脚本文件，如： JavaScript。


## HTML 样式- CSS

CSS (Cascading Style Sheets) 用于渲染HTML元素标签的样式。


### 如何使用CSS

CSS 是在 HTML 4 开始使用的,是为了更好的渲染HTML元素而引入的.

CSS 可以通过以下方式添加到HTML中:

- 内联样式- 在HTML元素中使用"style" **属性**
- 内部样式表 -在HTML文档头部 \<head\> 区域使用\<style\> **元素** 来包含CSS
- 外部引用 - 使用外部 CSS **文件**

最好的方式是通过外部引用CSS文件


### HTML样式实例 - 背景颜色

背景色属性（background-color）定义一个元素的背景颜色：

```html
<body style="background-color:yellow;">
<h2 style="background-color:red;">这是一个标题</h2>
<p style="background-color:green;">这是一个段落。</p>
</body>
```


### HTML 样式实例 - 字体, 字体颜色 ，字体大小

我们可以使用font-family（字体），color（颜色），和font-size（字体大小）属性来定义字体的样式:

```html
<h1 style="font-family:verdana;">一个标题</h1>
<p style="font-family:arial;color:red;font-size:20px;">一个段落。</p>
```


### HTML 样式实例 - 文本对齐方式

使用 text-align（文字对齐）属性指定文本的水平与垂直对齐方式：

```html
<h1 style="text-align:center;">居中对齐的标题</h1>
<p>这是一个段落。</p>
```


### 内部样式表

当单个文件需要特别样式时，就可以使用内部样式表。你可以在\<head\> 部分通过 \<style\>标签定义内部样式表

```html
<head>
<style type="text/css">
body {background-color:yellow;}
p {color:blue;}
</style>
</head>
```


### 外部样式表

当样式需要被应用到很多页面的时候，外部样式表将是理想的选择。使用外部样式表，你就可以通过更改一个文件来改变整个站点的外观。

```html
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```


### 注

css想要自学可以直接看这个[CSS 教程 | 菜鸟教程](https://www.runoob.com/css/css-tutorial.html)


## HTML 图像- 图像标签（ \<img\>）和源属性（Src）

在 HTML 中，图像由\<img\> 标签定义。

\<img\> 是空标签，意思是说，它只包含属性，并且没有闭合标签。

要在页面上显示图像，你需要使用源属性（src）。src 指 "source"。源属性的值是图像的 URL 地址。

**定义图像的语法是：**

\<img src\="*url*" alt\="*some_text*"\>`

URL 指存储图像的位置。如果名为 "pulpit.jpg" 的图像位于 www.runoob.com 的 images 目录中，那么其 URL 为 [http://www.runoob.com/images/pulpit.jpg](https://www.runoob.com/images/pulpit.jpg)。

浏览器将图像显示在文档中图像标签出现的地方。如果你将图像标签置于两个段落之间，那么浏览器会首先显示第一个段落，然后显示图片，最后显示第二段。


## HTML 图像- Alt属性

alt 属性用来为图像定义一串预备的可替换的文本。

替换文本属性的值是用户定义的。

\<img src\="boat.gif" alt\="Big Boat"\>

在浏览器无法载入图像时，替换文本属性告诉读者她们失去的信息。此时，浏览器将显示这个替代性的文本而不是图像。为页面上的图像都加上替换文本属性是个好习惯，这样有助于更好的显示信息，并且对于那些使用纯文本浏览器的人来说是非常有用的。


## HTML 图像- 设置图像的高度与宽度

height（高度） 与 width（宽度）属性用于设置图像的高度与宽度。

属性值默认单位为像素:

\<img src\="pulpit.jpg" alt\="Pulpit rock" width\="304" height\="228"\>

**提示:**  指定图像的高度和宽度是一个很好的习惯。如果图像指定了高度宽度，页面加载时就会保留指定的尺寸。如果没有指定图片的大小，加载页面时有可能会破坏HTML页面的整体布局。


### 注

如果没能设置正确的位置，则会显示出一个破碎的图像