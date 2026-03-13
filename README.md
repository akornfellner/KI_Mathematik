# Reveal.js Presentation Template

This template is markdown-first. In normal use, you only change:

- `presentation.yml`
- `slides/slides.md` — title slide
- `slides/topics/*.md` — one file per subtopic

## Local preview

For live reload, use:

```bash
just present
```

Then open `http://localhost:1948`.

If you only want a plain static server, any static file server works. Example:

```bash
python3 -m http.server 1948
```

Then open `http://localhost:1948`.

## Config

`presentation.yml`

```yaml
title: Slides
lang: de
footer-center: Your Name
theme: dark
highlightTheme: monokai
```

Supported themes:

- `dark`
- `white`
- `hak` — white base, red headings, HAK logo
- `htl` — white base, blue headings, HTL logo

Supported highlight themes:

- `monokai`
- `github`
- `atomOneLight`
- `atomOneDark`

## Logos

The `hak` and `htl` themes automatically display their respective logo (`hak.png` / `htl.png`) in the bottom-left corner. No logo is shown for other themes.

## Slide structure

```
slides/
  slides.md              ← title slide (first slide of the presentation)
  topics/
    01-intro.md          ← first subtopic
    02-details.md        ← second subtopic
    ...
```

- **`slides/slides.md`** contains only the title/header slide.
- Each `.md` file in **`slides/topics/`** becomes a new horizontal section (navigate left/right between topics).
- Files are loaded in filename order — use a numeric prefix (e.g. `01-`, `02-`) to control the sequence.
- Within a topic file, use `----` to separate vertical slides (navigate up/down within a topic).
