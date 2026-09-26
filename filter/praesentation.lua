-- Baut aus dem Block `vortrag:` in _quarto.yml und der gewählten Brand-Datei:
--   * Folien:  die Titelfolie (Titel, Logo, Untertitel, Autor, Link)
--   * Handout: Titelseite, Kopfzeile ab Seite 2 und Inhaltsverzeichnis
-- Beim dunklen Design wird das Handout trotzdem weiß gesetzt.

local TEXTE = {
  de = { handout = "Begleitskript zum Vortrag", folien = "Folien", stand = "Stand", inhalt = "Inhalt" },
  en = { handout = "Lecture notes", folien = "Slides", stand = "Version", inhalt = "Contents" },
}

local stringify = pandoc.utils.stringify

-- Brand-Datei lesen (YAML über den Markdown-Reader von Pandoc parsen)
local function brand_lesen(pfad)
  local datei = io.open(pfad, "r")
  if not datei then
    return nil
  end
  local inhalt = datei:read("a")
  datei:close()
  return pandoc.read("---\n" .. inhalt .. "\n---\n", "markdown").meta
end

local function palette_aufloesen(brand, wert)
  local palette = brand.color and brand.color.palette or {}
  local seen = 0
  while wert and palette[wert] and seen < 10 do
    wert = stringify(palette[wert])
    seen = seen + 1
  end
  return wert
end

local function ist_dunkel(hex)
  if not hex or not hex:match("^#%x%x%x%x%x%x$") then
    return false
  end
  local r = tonumber(hex:sub(2, 3), 16) / 255
  local g = tonumber(hex:sub(4, 5), 16) / 255
  local b = tonumber(hex:sub(6, 7), 16) / 255
  return 0.299 * r + 0.587 * g + 0.114 * b < 0.5
end

-- Pfad des großen Logos (relativ zum Projekt) oder nil
local function logo_pfad(brand, brand_pfad)
  if not brand.logo then
    return nil
  end
  local name = brand.logo.large or brand.logo.medium or brand.logo.small
  if not name then
    return nil
  end
  name = stringify(name)
  local bilder = brand.logo.images or {}
  local datei = bilder[name] and stringify(bilder[name]) or name
  return pandoc.path.join({ pandoc.path.directory(brand_pfad), datei })
end

-- Meta-Wert als Typst-Markup
local function typst(wert)
  if wert == nil then
    return ""
  end
  local inlines = pandoc.Inlines(wert)
  local text = pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), "typst")
  return (text:gsub("%s+$", ""))
end

local function titelfolie(v, logo)
  local bloecke = pandoc.Blocks({})
  local titel = pandoc.Header(2, pandoc.Inlines(v.titel or ""), pandoc.Attr("", { "titelfolie" }))
  bloecke:insert(titel)
  if logo then
    bloecke:insert(pandoc.Para({ pandoc.Image({}, logo, "", pandoc.Attr("", { "titel-logo", "nostretch" })) }))
  end
  if v.untertitel then
    bloecke:insert(pandoc.Para({ pandoc.Span(pandoc.Inlines(v.untertitel), pandoc.Attr("", { "untertitel" })) }))
  end
  if v.autor then
    bloecke:insert(pandoc.Para({ pandoc.Span(pandoc.Inlines(v.autor), pandoc.Attr("", { "autor" })) }))
  end
  if v.url then
    local url = stringify(v.url)
    bloecke:insert(pandoc.Div({ pandoc.Para({ pandoc.Link(url, url) }) }, pandoc.Attr("", { "quelle" })))
  end
  return bloecke
end

local function handout_anfang(v, logo, texte, dunkel)
  local titel = typst(v.titel)
  local autor = typst(v.autor)
  local untertitel = v.untertitel and typst(v.untertitel) or texte.handout
  local kopf_text = titel .. (autor ~= "" and (" · " .. autor) or "")

  local logo_gross = logo
      and ('#box(width: 70%, height: 4.5cm, image("/' .. logo .. '", width: 100%, height: 100%, fit: "contain"))\n#v(2.5cm)')
    or "#v(4cm)"
  local logo_klein = logo and ('box(height: 0.75cm, image("/' .. logo .. '", height: 100%))') or "[]"
  local link = ""
  if v.url then
    local url = stringify(v.url)
    local anzeige = url:gsub("^https?://", ""):gsub("/$", "")
    link = '#v(0.3em)\n#text(size: 10pt, fill: luma(110))[' .. texte.folien .. ': #link("' .. url .. '")[' .. anzeige .. "]]"
  end

  local dunkel_regeln = ""
  if dunkel then
    -- Dunkles Folien-Design: Handout trotzdem hell und druckbar
    dunkel_regeln = [[
#let brand-color = brand-color + (background: white, foreground: luma(20))
#let brand-color-background = brand-color.pairs().fold((:), (acc, (k, c)) => acc + ((k): color.mix((c, 15%), (white, 85%))))
#set page(fill: white)
#set text(fill: luma(20))
#show heading: set text(fill: brand-color.primary.darken(25%))
#show link: set text(fill: brand-color.primary.darken(25%))
]]
  end

  return dunkel_regeln .. [[
#set page(
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 8.5pt, fill: luma(120))
      grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        ]] .. logo_klein .. [[,
        []] .. kopf_text .. [[],
      )
      v(-0.35em)
      line(length: 100%, stroke: 0.4pt + luma(200))
    }
  },
)
#page(numbering: none)[
  #v(2.5cm)
  #align(center)[
    ]] .. logo_gross .. [[

    #text(size: 30pt, weight: 700, fill: ]] .. (dunkel and "brand-color.primary.darken(25%)" or "brand-color.primary") .. [[)[]] .. titel .. [[]
    #v(0.4em)
    #text(size: 15pt, fill: luma(90))[]] .. untertitel .. [[]
    #v(2cm)
    #text(size: 13pt)[]] .. autor .. [[]
    ]] .. link .. [[

  ]
  #v(1fr)
  #align(center, text(size: 9pt, fill: luma(130))[]] .. texte.stand .. [[: #datetime.today().display("[day].[month].[year]")])
]
#outline(title: []] .. texte.inhalt .. [[], depth: 1)
]]
end

function Pandoc(doc)
  local v = doc.meta.vortrag or {}
  local lang = stringify(doc.meta.lang or "de"):sub(1, 2)
  local texte = TEXTE[lang] or TEXTE.de

  local brand_pfad = doc.meta.brand and stringify(doc.meta.brand) or nil
  local brand = brand_pfad and brand_lesen(brand_pfad) or {}
  local logo = brand_pfad and logo_pfad(brand, brand_pfad) or nil
  local hintergrund = brand.color and brand.color.background and palette_aufloesen(brand, stringify(brand.color.background))
  local dunkel = ist_dunkel(hintergrund)

  if quarto.doc.is_format("revealjs") then
    local neu = titelfolie(v, logo)
    neu:extend(doc.blocks)
    doc.blocks = neu
  elseif quarto.doc.is_format("typst") then
    local neu = pandoc.Blocks({ pandoc.RawBlock("typst", handout_anfang(v, logo, texte, dunkel)) })
    neu:extend(doc.blocks)
    doc.blocks = neu
  end
  return doc
end
