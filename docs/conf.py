# Configuration file for the Sphinx documentation builder.

project = 'Likelihood-Based Inference'
author = 'Kevin Korfmann'
copyright = '2026, Kevin Korfmann'
release = '1.0'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Likelihood-Based Inference: From Foundations to Research'

html_theme_options = {
    "light_css_variables": {
        # Warm off-white backgrounds — easy on the eyes
        "color-background-primary": "#faf8f5",
        "color-background-secondary": "#f3efe9",
        "color-background-hover": "#ece7df",
        "color-background-hover--transparent": "#ece7df00",
        "color-background-border": "#e0d9cf",
        "color-background-item": "#ede9e2",

        # Soft dark text — not pure black
        "color-foreground-primary": "#3b3632",
        "color-foreground-secondary": "#63594f",
        "color-foreground-muted": "#8c8078",
        "color-foreground-border": "#d4ccc2",

        # Muted teal accent — calming and readable
        "color-brand-primary": "#4a7c72",
        "color-brand-content": "#3d6e64",

        # Sidebar
        "color-sidebar-background": "#f3efe9",
        "color-sidebar-background-border": "#e0d9cf",
        "color-sidebar-brand-text": "#3b3632",
        "color-sidebar-caption-text": "#63594f",
        "color-sidebar-link-text": "#3b3632",
        "color-sidebar-link-text--top-level": "#3b3632",
        "color-sidebar-item-background--current": "#ece7df",
        "color-sidebar-item-background--hover": "#ece7df",
        "color-sidebar-item-expander-color": "#8c8078",
        "color-sidebar-item-expander-color--hover": "#63594f",
        "color-sidebar-search-background": "#faf8f5",
        "color-sidebar-search-background--focus": "#ffffff",

        # Admonitions
        "color-admonition-background": "#f3efe9",

        # API / code
        "color-api-background": "#f3efe9",
        "color-api-background-hover": "#ece7df",
        "color-highlight-on-target": "#fdf3dc",
        "color-inline-code-background": "#f0ebe4",

        # Cards / header / announcement
        "color-card-background": "#faf8f5",
        "color-card-border": "#e0d9cf",
        "color-header-background": "#f3efe9",
        "color-header-border": "#e0d9cf",
    },
    # Force light mode only — no dark toggle
    "dark_css_variables": {
        "color-background-primary": "#faf8f5",
        "color-background-secondary": "#f3efe9",
        "color-background-hover": "#ece7df",
        "color-background-hover--transparent": "#ece7df00",
        "color-background-border": "#e0d9cf",
        "color-background-item": "#ede9e2",
        "color-foreground-primary": "#3b3632",
        "color-foreground-secondary": "#63594f",
        "color-foreground-muted": "#8c8078",
        "color-foreground-border": "#d4ccc2",
        "color-brand-primary": "#4a7c72",
        "color-brand-content": "#3d6e64",
        "color-sidebar-background": "#f3efe9",
        "color-sidebar-background-border": "#e0d9cf",
        "color-sidebar-brand-text": "#3b3632",
        "color-sidebar-caption-text": "#63594f",
        "color-sidebar-link-text": "#3b3632",
        "color-sidebar-link-text--top-level": "#3b3632",
        "color-sidebar-item-background--current": "#ece7df",
        "color-sidebar-item-background--hover": "#ece7df",
        "color-sidebar-item-expander-color": "#8c8078",
        "color-sidebar-item-expander-color--hover": "#63594f",
        "color-sidebar-search-background": "#faf8f5",
        "color-sidebar-search-background--focus": "#ffffff",
        "color-admonition-background": "#f3efe9",
        "color-api-background": "#f3efe9",
        "color-api-background-hover": "#ece7df",
        "color-highlight-on-target": "#fdf3dc",
        "color-inline-code-background": "#f0ebe4",
        "color-card-background": "#faf8f5",
        "color-card-border": "#e0d9cf",
        "color-header-background": "#f3efe9",
        "color-header-border": "#e0d9cf",
    },
    "light_logo": "logo.jpg",
    "dark_logo": "logo.jpg",
    "sidebar_hide_name": False,
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/donate.html",
    ],
}

# -- LaTeX output ------------------------------------------------------------
latex_engine = 'xelatex'

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath,amssymb,amsthm}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{eso-pic}
\usepackage{xcolor}
\usepackage{tikz}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
% Let hyperref handle math in section titles gracefully
\pdfstringdefDisableCommands{%
  \renewcommand{\boldsymbol}[1]{#1}%
  \renewcommand{\bm}[1]{#1}%
  \let\mathbb\@gobble
}
''',
    'maketitle': r'''
\clearpage
\thispagestyle{empty}
% --- Full-page background image via eso-pic (xelatex-safe) ---
\AddToShipoutPictureBG*{%
  \AtPageLowerLeft{%
    \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{cover.jpg}%
  }%
}
% --- Dark overlay for text legibility ---
\AddToShipoutPictureBG*{%
  \AtPageLowerLeft{%
    \begin{tikzpicture}
      \fill[black,opacity=0.58] (0pt,0pt) rectangle (\paperwidth,\paperheight);
    \end{tikzpicture}%
  }%
}
% --- Cover content ---
\color{white}
\null
\vspace{4.5cm}
\begin{center}
  {\fontsize{46}{56}\selectfont\bfseries Likelihood-Based\par}
  \vspace{0.25cm}
  {\fontsize{46}{56}\selectfont\bfseries Inference\par}
  \vspace{1.2cm}
  {\color{white!40}\rule{8cm}{0.5pt}\par}
  \vspace{1.2cm}
  {\fontsize{18}{24}\selectfont\color{white!80} From Foundations to Research\par}
  \vspace{1.4cm}
  {\fontsize{13}{18}\selectfont\color{white!60}\textsc{Kevin Korfmann}\par}
\end{center}
\vfill
\begin{center}
  {\fontsize{9}{12}\selectfont\color{white!35}
   A comprehensive guide from basic probability through research-grade algorithms\par}
  \vspace{0.6cm}
  {\fontsize{9}{12}\selectfont\color{white!25} 2026\par}
\end{center}
\vspace{1.8cm}
\clearpage
''',
    'tableofcontents': r'\tableofcontents',
}

latex_documents = [
    ('index', 'LikelihoodInference.tex',
     'Likelihood-Based Inference',
     'Kevin Korfmann', 'manual'),
]

latex_additional_files = ['_static/cover.jpg']

html_css_files = ['custom.css']

# -- Extension configuration -------------------------------------------------
todo_include_todos = True
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

numfig = True
math_numfig = True
numfig_secnum_depth = 2
