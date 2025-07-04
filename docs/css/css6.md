## CSS 图像拼合技术

[CSS 图像拼合技术 | 菜鸟教程](https://www.runoob.com/css/css-image-sprites.html)

## CSS 媒体类型

[CSS 媒体类型 | 菜鸟教程](https://www.runoob.com/css/css-mediatypes.html)

## CSS 属性 选择器

CSS 属性选择器用于根据元素的属性或属性值来选择 HTML 元素。

属性选择器可以帮助你在不需要为元素添加类或 ID 的情况下对其进行样式化。

**注意：** IE7 和 IE8 需声明 !DOCTYPE 才支持属性选择器！IE6 和更低的版本不支持属性选择器。

### 常见的 CSS 属性选择器：

#### 1、[attribute]

选择带有指定属性的所有元素（无论属性值是什么）。

```css
/* 选择所有具有 `type` 属性的元素 */
[type] {
  border: 1px solid red;
}
```

#### 2、[attribute\="value"]

选择具有指定属性和值完全匹配的元素。

```css
/* 选择所有 `type` 属性等于 `text` 的元素 */
[type="text"] {
  background-color: yellow;
}
```

#### 3、[attribute\~\="value"]

选择属性值中包含指定词（用空格分隔的词列表之一）的元素。

```css
/* 选择属性值中包含 `button` 的元素 */
[class~="button"] {
  font-weight: bold;
}
```

#### 4、[attribute|\="value"]

选择具有指定值或者以指定值开头并紧跟着连字符 - 的属性值的元素，常用于语言代码。

```css
/* 选择所有 `lang` 属性是 `en` 或者以 `en-` 开头的元素 */
[lang|="en"] {
  color: green;
}
```

#### 5、[attribute\^\="value"]

选择属性值以指定值开头的元素。

```css
/* 选择所有 `href` 属性以 `https` 开头的链接 */
[href^="https"] {
  text-decoration: none;
}
```

#### 6、[attribute\$\="value"]

选择属性值以指定值结尾的元素。

```css
/* 选择所有 src 属性以 .jpg 结尾的图片 */
[src$=".jpg"] {
  border: 2px solid blue;
}
```

#### 7、[attribute\*\="value"]

选择属性值中包含指定值的元素。

```css
/* 选择所有 `title` 属性中包含 `tutorial` 的元素 */
[title*="tutorial"] {
  font-style: italic;
}
```

通过属性选择器，你可以轻松选择元素并基于它们的属性设置特定样式，增加了灵活性和可维护性。

### 属性选择器

下面的例子是把包含标题（title）的所有元素变为蓝色：

```css
[title]
{
    color:blue;
}
```

### 属性和值选择器

下面的实例改变了标题title\='runoob'元素的边框样式:

```css
[title=runoob]
{
    border:5px solid green;
}
```

### 属性和值的选择器 - 多值

下面是包含指定值的title属性的元素样式的例子，使用（\~）分隔属性和值:

```css
[title~=hello] { color:blue; }
```

下面是包含指定值的lang属性的元素样式的例子，使用（|）分隔属性和值:

```css
[lang|=en] { color:blue; }
```

### 表单样式

属性选择器样式无需使用class或id的形式:

```css
input[type="text"]
{
    width:150px;
    display:block;
    margin-bottom:10px;
    background-color:yellow;
}
input[type="button"]
{
    width:120px;
    margin-left:35px;
    display:block;
}
```

## CSS 表单

### 输入框(input) 样式

使用 width 属性来设置输入框的宽度：

```css
input {
  width: 100%;
}
```

以上实例中设置了所有 \<input\> 元素的宽度为 100%，如果你只想设置指定类型的输入框可以使用以下属性选择器：

- `input[type=text]` - 选取文本输入框
- `input[type=password]` - 选择密码的输入框
- `input[type=number]` - 选择数字的输入框
- ...

### 输入框填充

使用 **padding** 属性可以在输入框中添加内边距。

```css
input[type=text] {
  width: 100%;
  padding: 12px 20px;
  margin: 8px 0;
  box-sizing: border-box;
}
```

注意我们设置了 `box-sizing` 属性为 `border-box`。这样可以确保浏览器呈现出带有指定宽度和高度的输入框是把边框和内边距一起计算进去的。
更多内容可以阅读 [CSS3 框大小](https://www.runoob.com/css3/css3-box-sizing.html) 。

### 输入框(input) 边框

使用 `border` 属性可以修改 input 边框的大小或颜色，使用 `border-radius` 属性可以给 input 添加圆角：

```css
input[type=text] {
  border: 2px solid red;
  border-radius: 4px;
}
```

如果你只想添加底部边框可以使用 border-bottom 属性:

```css
input[type=text] {
  border: none;
  border-bottom: 2px solid red;
}
```

### 输入框(input) 颜色

可以使用 `background-color` 属性来设置输入框的背景颜色，`color` 属性用于修改文本颜色：

```css
input[type=text] {
  background-color: #3CBC8D;
  color: white;
}
```

### 输入框(input) 聚焦

默认情况下，一些浏览器在输入框获取焦点时（点击输入框）会有一个蓝色轮廓。我们可以设置 input 样式为 `outline: none;` 来忽略该效果。

使用 `:focus` 选择器可以设置输入框在获取焦点时的样式：

```css
input[type=text]:focus {
  background-color: lightblue;
}  /*这个显示的效果是鼠标放上去之后会有淡蓝色的背景*/
```

```css
input[type=text]:focus {
  border: 3px solid #555;
}  /*这个显示的效果是鼠标放上去之后仍旧保持白色背景*/
```

### 输入框(input) 图标

如果你想在输入框中添加图标，可以使用 `background-image` 属性和用于定位的`background-position` 属性。注意设置图标的左边距，让图标有一定的空间：

```css
input[type=text] {
  background-color: white;
  background-image: url('searchicon.png');
  background-position: 10px 10px; 
  background-repeat: no-repeat;
  padding-left: 40px;
} /*呈现效果是文本框最左侧有一个搜索的图标*/
```

### 带动画的搜索框

以下实例使用了 CSS `transition` 属性，该属性设置了输入框在获取焦点时会向右延展。你可以在 [CSS 动画](https://www.runoob.com/css3/css3-animations.html) 章节查看更多内容。

```css
input[type=text] {
  -webkit-transition: width 0.4s ease-in-out;
  transition: width 0.4s ease-in-out;
}
 
input[type=text]:focus {
  width: 100%;
}
```

### 文本框（textarea）样式

**注意:**  使用 `resize` 属性来禁用文本框可以重置大小的功能（一般拖动右下角可以重置大小）。

```css
textarea {
  width: 100%;
  height: 150px;
  padding: 12px 20px;
  box-sizing: border-box;
  border: 2px solid #ccc;
  border-radius: 4px;
  background-color: #f8f8f8;
  resize: none;
}
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_form_textarea)

### 下拉菜单（select）样式

```css
select {
  width: 100%;
  padding: 16px 20px;
  border: none;
  border-radius: 4px;
  background-color: #f1f1f1;
}
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_form_select)

### 按钮样式

```css
input[type=button], input[type=submit], input[type=reset] {
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 16px 32px;
  text-decoration: none;
  margin: 4px 2px;
  cursor: pointer;
}
 
/* 提示: 使用 width: 100% 设置全宽按钮 */
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_form_button)

更多内容可以参考我们的 [CSS 按钮](https://www.runoob.com/css3/css3-buttons.html) 教程。

### 响应式表单

响应式表单可以根据浏览器窗口的大小重新布局各个元素，我们可以通过重置浏览器窗口大小来查看效果：

**高级:**  以下实例使用了[CSS3 多媒体查询](https://www.runoob.com/css3/css3-mediaqueries.html) 来创建一个响应式表单。

```css
* {
  box-sizing: border-box;
}
 
input[type=text], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
}
 
label {
  padding: 12px 12px 12px 0;
  display: inline-block;
}
 
input[type=submit] {
  background-color: #4CAF50;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  float: right;
}
 
input[type=submit]:hover {
  background-color: #45a049;
}
 
.container {
  border-radius: 5px;
  background-color: #f2f2f2;
  padding: 20px;
}
 
.col-25 {
  float: left;
  width: 25%;
  margin-top: 6px;
}
 
.col-75 {
  float: left;
  width: 75%;
  margin-top: 6px;
}
 
/* 清除浮动 */
.row:after {
  content: "";
  display: table;
  clear: both;
}
 
/* 响应式布局 layout - 在屏幕宽度小于 600px 时， 设置为上下堆叠元素 */
@media screen and (max-width: 600px) {
  .col-25, .col-75, input[type=submit] {
    width: 100%;
    margin-top: 0;
  }
}
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_form_responsive)

## CSS 计数器

CSS 计数器通过一个变量来设置，根据规则递增变量。

### 使用计数器自动编号

CSS 计数器根据规则来递增变量。

CSS 计数器使用到以下几个属性：

- `counter-reset` - 创建或者重置计数器
- `counter-increment` - 递增变量
- `content` - 插入生成的内容
- `counter()` 或 `counters()` 函数 - 将计数器的值添加到元素

要使用 CSS 计数器，得先用 counter-reset 创建：

以下实例在页面创建一个计数器 (在 body 选择器中)，每个 \<h2\> 元素的计数值都会递增，并在每个 \<h2\> 元素前添加 "Section \<*计数值*\>:"

```css
body {
  counter-reset: section;
}
 
h2::before {
  counter-increment: section;
  content: "Section " counter(section) ": ";
}
```

### 嵌套计数器

以下实例在页面创建一个计数器，在每一个 \<h1\> 元素前添加计数值 "Section \<*主标题计数值*\>.", 嵌套的计数值则放在 \<h2\> 元素的前面，内容为 "\<*主标题计数值*\>.\<*副标题计数值*\>":

```css
body {
  counter-reset: section;
}
 
h1 {
  counter-reset: subsection;
}
 
h1::before {
  counter-increment: section;
  content: "Section " counter(section) ". ";
}
 
h2::before {
  counter-increment: subsection;
  content: counter(section) "." counter(subsection) " ";
}
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_counters2)

计数器也可用于列表中，列表的子元素会自动创建。这里我们使用了 counters() 函数在不同的嵌套层级中插入字符串:

```css
ol {
  counter-reset: section;
  list-style-type: none;
}
 
li::before {
  counter-increment: section;
  content: counters(section,".") " ";
}
```

[菜鸟教程在线编辑器](https://www.runoob.com/try/try.php?filename=trycss_counters3)

### CSS 计数器属性

| 属性 | 描述                                                |
| ------ | ----------------------------------------------------- |
| [content](https://www.runoob.com/cssref/pr-gen-content.html)     | 使用 ::before 和 ::after 伪元素来插入自动生成的内容 |
| [counter-increment](https://www.runoob.com/cssref/pr-gen-counter-increment.html)     | 递增一个或多个值                                    |
| [counter-reset](https://www.runoob.com/cssref/pr-gen-counter-reset.html)     | 创建或重置一个或多个计数器                          |

## CSS 网页布局

[CSS 网页布局 | 菜鸟教程](https://www.runoob.com/css/css-website-layout.html)

## CSS !important 规则

### 什么是 !important

CSS 中的 !important 规则用于增加样式的权重。

!important 与优先级无关，但它与最终的结果直接相关，使用一个 !important 规则时，此声明将覆盖任何其他声明。

```css
#myid {
  background-color: blue;
}
 
.myclass {
  background-color: gray;
}
 
p {
  background-color: red !important;
}
```

以上实例中，尽管 ID 选择器和类选择器具有更高的优先级，但三个段落背景颜色都显示为红色，因为 !important 规则会覆盖 background-color 属性。

### 重要说明

使用 !important 是一个坏习惯，应该尽量避免，因为这破坏了样式表中的固有的级联规则 使得调试找 bug 变得更加困难了。

当两条相互冲突的带有 !important 规则的声明被应用到相同的元素上时，拥有更大优先级的声明将会被采用。

**使用建议：**

- **一定**要优先考虑使用样式规则的优先级来解决问题而不是 `!important`
- **只有**在需要覆盖全站或外部 CSS 的特定页面中使用 `!important`
- **永远不要**在你的插件中使用 `!important`
- **永远不要**在全站范围的 CSS 代码中使用 `!important`

### 何时使用 !important

如果要在你的网站上设定一个全站样式的 CSS 样式可以使用 !important。

## CSS 实例

[CSS 实例 | 菜鸟教程](https://www.runoob.com/css/css-examples.html)到这里面自行搜索吧