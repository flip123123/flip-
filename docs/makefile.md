## Makefile 规则

target ... : prerequisites ...

command

...

...

- target是一个目标文件，可以是Object File，也可以是执行文件。还可以是一个标签（Label）
- prerequisites是，要生成那个target所需要的文件或是目标。
- command是make需要执行的命令

也就是说，target这一个或多个的目标文件依赖于prerequisites中的文件，其生成规则定义在command中。 说白一点就是说，**prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行，** 这就是 Makefile的规则。也就是Makefile中最核心的内容。

### 一个示例：

```makefile
edit : main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o
cc -o edit main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o
main.o : main.c defs.h
cc -c main.c
kbd.o : kbd.c defs.h command.h
cc -c kbd.c
command.o : command.c defs.h command.h
cc -c command.c
display.o : display.c defs.h buffer.h
cc -c display.c
insert.o : insert.c defs.h buffer.h
cc -c insert.c
search.o : search.c defs.h buffer.h
cc -c search.c
files.o : files.c defs.h buffer.h command.h
cc -c files.c
utils.o : utils.c defs.h
cc -c utils.c
clean :
rm edit main.o kbd.o command.o display.o \
insert.o search.o files.o utils.ocommand.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h
.PHONY : clean
clean :
rm edit $(objects)
```


现在我们逐行来分析

#### **第一部分：目标文件和最终可执行文件的定义与构建规则**

```makefile
edit : main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o
	cc -o edit main.o kbd.o command.o display.o \
	insert.o search.o files.o utils.o
```

- **`edit`**: 这是规则的**目标 (Target)** ，表示我们要构建一个名为edit的文件
- **`main.o kbd.o command.o display.o \`** : 这是目标 `edit` 的**依赖项 (Prerequisites)** 。这些是 `.o` 文件，表示它们是编译过程中生成的目标代码文件。反斜杠 `\` 表示这一行延续到下一行。

**整体含义**: 这条规则说明，要生成名为 `edit` 的可执行文件，必须先有 `main.o`, `kbd.o`, `command.o`, `display.o`, `insert.o`, `search.o`, `files.o`, 和 `utils.o` 这几个目标文件

- **`cc`**: 这是要执行的**命令 (Command)** 。`cc` 通常是 C 语言编译器的命令（在很多系统上，`cc` 是 `gcc` 的一个符号链接或别名）。
-  **`-o edit`**: 这是编译器的选项，表示将链接的结果输出到名为 `edit` 的文件中。
- **`main.o kbd.o command.o display.o \`** : 这是链接器需要的输入文件，即所有编译好的目标文件

**整体含义**: 如果上述所有 `.o` 文件都存在（并且比 `edit` 文件新，或者 `edit` 不存在），则执行此命令来将所有 `.o` 文件链接成一个名为 `edit` 的可执行程序。这是**链接 (Linking)**  步骤。


#### 第二部分：各个 `.o` 目标文件的编译规则

使用了 **隐式规则** 的语法格式

```makefile
main.o : main.c defs.h
cc -c main.c
kbd.o : kbd.c defs.h command.h
cc -c kbd.c
command.o : command.c defs.h command.h
cc -c command.c
display.o : display.c defs.h buffer.h
cc -c display.c
insert.o : insert.c defs.h buffer.h
cc -c insert.c
search.o : search.c defs.h buffer.h
cc -c search.c
files.o : files.c defs.h buffer.h command.h
cc -c files.c
utils.o : utils.c defs.h
cc -c utils.c
```

这一块就是在指明，每一个中间文件所依赖的源文件和头文件

cc指的是c编译器，-c 只进行编译不进行链接，比如main,c会生成main.o

#### 第三部分：清理规则

```makefile
clean :
	rm edit main.o kbd.o command.o display.o \
	insert.o search.o files.o utils.o
```

- **`clean :`** :

  - **目标**: `clean`。这是一个**伪目标 (Phony Target)** ，因为它本身并不对应一个实际存在的文件。它代表一个动作。
  - **依赖项**: 没有明确列出依赖项。
  - **意义**: 当你执行 `make clean` 命令时，会触发这个规则。
- **`rm edit main.o kbd.o command.o display.o \`**

  - **`rm`**: 这是 **remove** 命令，用于删除文件。
  - **`edit main.o kbd.o command.o display.o \`** : 这是要删除的文件列表。反斜杠表示延续。
  - **`insert.o search.o files.o utils.o`**: 这是要删除文件的列表延续。
  - **意义**: 清除所有在构建过程中生成的文件（最终的可执行文件 `edit` 和所有的目标文件 `.o`）

### Makefile里有什么？

Makefile里主要包含了五个东西：显式规则、隐晦规则、变量定义、文件指示和注释。

1、显式规则。显式规则说明了，如何生成一个或多的的目标文件。这是由Makefile的书写者明显指出，要生成的文件，文件的依赖文件，生成的命令。

2、隐晦规则。由于我们的make有自动推导的功能，所以隐晦的规则可以让我们比较粗糙地简略地书写Makefile，这是由make所支持的。

3、变量的定义。在Makefile中我们要定义一系列的变量，变量一般都是字符串，这个有点你C语言中的宏，当Makefile被执行时，其中的变量都会被扩展到相应的引用位置上。

4、文件指示。其包括了三个部分，一个是在一个Makefile中引用另一个Makefile，就像C语言中的include一样；另一个是指根据某些情 况指定Makefile中的有效部分，

就像C语言中的预编译#if一样；还有就是定义一个多行的命令。有关这一部分的内容，我会在后续的部分中讲述。

5、注释。Makefile中只有行注释，和UNIX的Shell脚本一样，其注释是用“#”字符，这个就像C/C++中的“//”一样。如果你要在你的Makefile中使用“[#”字符，可以用反斜框进行转义，如：](http://tieba.baidu.com/hottopic/browse/hottopic?topic_id=0&topic_name=%E2%80%9D%E5%AD%97%E7%AC%A6%EF%BC%8C%E5%8F%AF%E4%BB%A5%E7%94%A8%E5%8F%8D%E6%96%9C%E6%A1%86%E8%BF%9B%E8%A1%8C%E8%BD%AC%E4%B9%89%EF%BC%8C%E5%A6%82%EF%BC%9A%E2%80%9C%5C&is_video_topic=0)`\#`。

### 工作方式

1. 读入所有的Makefile。
2. 读入被include的其它Makefile。
3. 初始化文件中的变量。
4. 推导隐晦规则，并分析所有规则。
5. 为所有的目标文件创建依赖关系链。
6. 根据依赖关系，决定哪些目标要重新生成。
7. 执行生成命令

1-5步为第一个阶段，6-7为第二个阶段。第一个阶段中，如果定义的变量被使用了，那么，make会把其展开在使用的位置。但make并不会完全马上展 开，make使用的是拖延战术，如果变量出现在依赖关系的规则中，那么仅当这条依赖被决定要使用了，变量才会在其内部展开。

### 书写规则

规则包含两个部分，一个是依赖关系，一个是生成目标的方法

在Makefile中，规则的顺序是很重要的，因为，Makefile中只应该有一个最终目标，其它的目标都是被这个目标所连带出来的，所以一定要让 make知道你的最终目标是什么。一般来说，定义在Makefile中的目标可能会有很多，但是第一条规则中的目标将被确立为最终的目标。如果第一条规则 中的目标有很多个，那么，第一个目标会成为最终的目标。make所完成的也就是这个目标。

#### 规则举例

```makefile
foo.o : foo.c defs.h # foo模块
cc -c -g foo.c
```

在这个例子中，foo.o是我们的目标，foo.c和defs.h是目标所依赖的源文件，而只有一个命令“cc -c -g foo.c”（以Tab键开头）。这个规则告诉我们两件事：

1. 文件的依赖关系，foo.o依赖于foo.c和defs.h的文件，如果foo.c和defs.h的文件日期要比foo.o文件日期要新，或是foo.o不存在，那么依赖关系发生
2. 如果生成（或更新）foo.o文件。也就是那个cc命令，其说明了，如何生成foo.o这个文件。（当然foo.c文件include了defs.h文件）


#### 规则的语法

targets : prerequisites
command
...
或是这样：
targets : prerequisites ; command
command
...

每一行的注释见开篇

注：如果命令太长，你可以使用反斜框（‘\’）作为换行符。make对一行上有多少个字符没有限制。

规则告诉make两件事，**文件的依赖关系和如何成成目标文件**

一般来说，make会以UNIX的标准Shell，也就是/bin/sh来执行命令。

#### 在规则中使用通配符

如果我们想定义一系列比较类似的文件，我们很自然地就想起使用通配符<sup>（通配符是一类特殊的字符，它们可以代表其他一个或多个字符。在计算机领域，通配符主要用于文件搜索、字符串匹配、数据库查询等场景中，用来简化输入和提高效率。 以下是两种最常见的通配符及其用法：  1. 在文件系统和命令行中使用 (Globbing) 在操作系统命令行中，我们经常使用通配符来匹配文件名或目录名。  * (星号):  代表零个或多个任意字符。 示例: .txt: 匹配所有以 .txt 结尾的文件，例如 document.txt, report.txt, .txt。 data: 匹配所有以 data 开头的文件，例如 data.csv, data_analysis.py, datamodel。 : 匹配当前目录下所有文件和目录（除了以 . 开头的隐藏文件）。 doc.*: 匹配所有以 doc 开头，后面跟着任意字符，再接着一个点，再接着任意字符的文件，例如 document.txt, doc.pdf, doc123.log。 ? (问号):  代表单个任意字符。 示例: report?.txt: 匹配 report1.txt, reportA.txt, reportX.txt，但不匹配 report10.txt 或 report.txt。 file_??.log: 匹配 file_01.log, file_AB.log，但不匹配 file_1.log 或 file_123.log。 [] (方括号):  代表方括号内任意一个字符集中的字符。 示例: [abc].txt: 匹配 a.txt, b.txt, c.txt。 file_[0-9].log: 匹配 file_0.log, file_1.log, …, file_9.log。这里 0-9 表示一个范围。 image_[01]_[abc].jpg: 匹配 image_0_a.jpg, image_0_b.jpg, image_0_c.jpg, image_1_a.jpg, image_1_b.jpg, image_1_c.jpg。 组合使用: [!abc].txt: 匹配所有不是 a, b, c 开头的文件（不常用，取决于 shell 的具体实现）。 file_[!0-9].log: 匹配文件名 file_ 后面不是数字的文件。 2. 在字符串匹配和SQL中使用 (Wildcard Characters) 在数据库查询（SQL）或一些文本处理工具中，也有类似的通配符，但符号可能有所不同。  SQL 中的 % (百分号):  代表零个或多个任意字符。 示例: SELECT * FROM customers WHERE name LIKE 'A%'; 匹配所有名字以 ‘A’ 开头的客户。 SELECT * FROM products WHERE code LIKE '%XYZ'; 匹配所有代码以 ‘XYZ’ 结尾的产品。 SELECT * FROM emails WHERE address LIKE '%@example.com'; 匹配所有域名为 ‘example.com’ 的邮箱地址。 SQL 中的 _ (下划线):  代表单个任意字符。 示例: SELECT * FROM employees WHERE employee_id LIKE 'EMP_001'; 匹配 EMP_A001, EMP_B001 等（这里假设 _ 匹配的是字母）。 SELECT * FROM items WHERE item_code LIKE 'P__123'; 匹配 PAB123, PXY123 等。 为什么需要通配符？ 简化操作: 无需输入所有文件名，就能对一组文件执行批量操作（如删除、复制、重命名）。 提高效率: 快速定位和筛选所需的文件或数据。 模式匹配: 用于搜索符合特定模式的字符串或数据。 理解和熟练使用通配符是进行高效文件管理和数据检索的关键。）</sup>。make支持三各通配符：“*”，“?”和“[...]”。这是和Unix的B-Shell是相同的。

波浪号（“”）字符在文件名中也有比较特殊的用途。如果是“/test”，这就表示当前用户的$HOME目录下的test目录。而 “~hchen/test”则表示用户hchen的宿主目录下的test目录。（这些都是Unix下的小知识了，make也支持）而在Windows或是 MS-DOS下，用户没有宿主目录，那么波浪号所指的目录则根据环境变量“HOME”而定

通配符代替了你一系列的文件，如“ *.c”表示所以后缀为c的文件。*

***一个需要我们注意的是，如果我们的文件名中有通配符，如：“*** **”，那么可以用转义字符“\”，如“*”来表示真实的“*”字符，而不是任意长度的字符串。**

#### 伪目标

在 Makefile 中，**伪目标 (Phony Targets)**  是指那些**不代表一个实际存在的文件名，而是代表一个特定的动作或命令集**。它们的主要目的是为了执行一些不直接与生成文件相关的操作，例如清理编译产物、运行测试、部署项目等。

##### **为什么需要伪目标？**

Makefile 的核心思想是根据文件之间的依赖关系来决定是否执行命令。例如，如果你有一个目标 `program`，它依赖于 `main.o` 和 `utils.o`，那么只有当 `main.o` 或 `utils.o` 比 `program` 旧时，Makefile 才会重新编译它们并链接成 `program`。

然而，很多 Makefile 中的操作并不是为了生成一个具体的物理文件。例如：

- `make clean`: 这个命令的目的是删除所有编译生成的文件（`.o` 文件，可执行文件等）。这些文件并不存在于 Makefile 本身中，而是构建过程的产物。如果你直接写 `clean:`，Makefile 会认为它是一个需要生成的文件，并且不会在 `clean` 目标比任何文件都“新”时执行。
- `make install`: 这个命令可能用于将编译好的程序复制到系统目录。它也不代表一个具体的文件。
- `make test`: 这个命令用于运行项目的测试套件。

如果不对这些“动作”目标进行特殊处理，Makefile 的一些默认行为可能会导致它们不按预期工作，或者在不应该执行时执行

因为，我们并不生成“clean”这个文件。“伪目标”并不是一个文件，只是一个标签，由于“伪目标”不是文件，所以make无法生成它的依赖关系和决定它是否要执行。我们只有通过显示地指明这个“目标”才能让其生效。当然，“伪目标”的取名不能和文件名重名，不然其就失去了“伪目标”的意义了。

当然，为了避免和文件重名的这种情况，我们可以使用一个特殊的标记“.PHONY”来显示地指明一个目标是“伪目标”，向make说明，不管是否有这个文件，这个目标就是“伪目标”。

```makefile
.PHONY : clean
```

只要有这个声明，不管是否有“clean”文件，要运行“clean”这个目标，只有“make clean”这样。于是整个过程可以这样写：

```makefile
.PHONY: clean
clean:
rm *.o temp
```

伪目标一般没有依赖的文件。但是，我们也可以为伪目标指定所依赖的文件。伪目标同样可以作为“默认目标”，只要将其放在第一个。一个示例就是，如果你的 Makefile需要一口气生成若干个可执行文件，但你只想简单地敲一个make完事，并且，所有的目标文件都写在一个Makefile中，那么你可以使 用“伪目标”这个特性：

```makefile
all : prog1 prog2 prog3
.PHONY : all
prog1 : prog1.o utils.o
cc -o prog1 prog1.o utils.o
prog2 : prog2.o
cc -o prog2 prog2.o
prog3 : prog3.o sort.o utils.o
cc -o prog3 prog3.o sort.o utils.o
```

由于Makefile中的第一个目标会被作为其默认目标。我们声明了一个“all”的伪目标，其依赖于其它三个目标。由于伪目标的特性是，总是被执行的<sup>（所以，当你执行 make 时：  make 查看 all。 make 查看 all 的依赖：prog1, prog2, prog3。 make 检查 prog1。如果 prog1.o 或 utils.o 比 prog1 新，make 会执行 cc -o prog1 prog1.o utils.o。 make 检查 prog2。如果 prog2.o 比 prog2 新，make 会执行 cc -o prog2 prog2.o。 make 检查 prog3。如果 prog3.o 或 sort.o 或 utils.o 比 prog3 新，make 会执行 cc -o prog3 prog3.o sort.o utils.o。 如果上述所有检查完成，并且所有必要的命令都已执行，那么 make 就认为 all 的目标已经达到。）</sup>，所以其依赖的那三个目标就总是不如“all”这个目标新。所以，其它三个目标的规则总是会被决议。也就达到了我们一口气生成多个目标的目的。 “.PHONY : all”声明了“all”这个目标为“伪目标”。

### 注

本篇是学习[Makefile详解（超级好）【mingw吧】_百度贴吧](https://tieba.baidu.com/p/591519800?see_lz=1)这篇文章，有不详明之处可以回去看原文