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
latex_engine = 'pdflatex'

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
''',
    'maketitle': r'''
\begin{titlepage}
\begin{tikzpicture}[remember picture, overlay]
  % --- Full-page cover image ---
  \node[inner sep=0pt] at (current page.center) {%
    \includegraphics[width=\paperwidth, height=\paperheight, keepaspectratio=false]{cover.jpg}%
  };
  % --- Dark gradient overlay for text legibility ---
  \fill[black, opacity=0.55] (current page.south west) rectangle (current page.north east);
  % --- Subtle top accent line ---
  \fill[white, opacity=0.15]
    ([yshift=-3.2cm]current page.north west) rectangle
    ([yshift=-3.25cm]current page.north east);
  % --- Title block ---
  \node[anchor=north, text width=14cm, align=center]
    at ([yshift=-4cm]current page.north) {%
      {\fontsize{42}{50}\selectfont\bfseries\color{white}%
       Likelihood-Based\\[6pt]Inference\par}%
    };
  % --- Subtitle ---
  \node[anchor=north, text width=14cm, align=center]
    at ([yshift=-8.2cm]current page.north) {%
      {\fontsize{18}{24}\selectfont\color{white!85}%
       From Foundations to Research\par}%
    };
  % --- Thin decorative rule ---
  \draw[white, opacity=0.4, line width=0.6pt]
    ([yshift=-9.4cm, xshift=-4cm]current page.north) --
    ([yshift=-9.4cm, xshift=4cm]current page.north);
  % --- Author ---
  \node[anchor=north, text width=14cm, align=center]
    at ([yshift=-10.2cm]current page.north) {%
      {\fontsize{14}{18}\selectfont\color{white!70}%
       Kevin Korfmann\par}%
    };
  % --- Bottom tagline ---
  \node[anchor=south, text width=14cm, align=center]
    at ([yshift=2cm]current page.south) {%
      {\fontsize{10}{14}\selectfont\color{white!50}%
       A comprehensive guide from basic probability\\
       through research-grade algorithms\par}%
    };
  % --- Year at very bottom ---
  \node[anchor=south, text width=14cm, align=center]
    at ([yshift=0.8cm]current page.south) {%
      {\fontsize{9}{12}\selectfont\color{white!40} 2026\par}%
    };
\end{tikzpicture}
\end{titlepage}
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
