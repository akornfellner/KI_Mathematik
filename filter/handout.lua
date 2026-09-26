-- Passt die Folieninhalte für das Handout (Typst/PDF) an:
--   * Überschriften mit {.slide-only} erscheinen nur auf den Folien
--   * `---` (Folientrenner ohne Titel) wird im Handout entfernt
--   * ::: notes (Sprechernotizen) wird im Handout zu normalem Fließtext
--   * ::: quelle wird im Handout klein und grau gesetzt
--   * „…“ als Typst-Anführungszeichen, damit Typst sie korrekt paart
--   * \left/\right in Formeln entfernen (Typst skaliert Klammern selbst;
--     sonst setzt Pandoc Leerzeichen und das Dezimalkomma bekommt Abstand)

if not quarto.doc.is_format("typst") then
  return {}
end

return {
  {
    Header = function(el)
      if el.classes:includes("slide-only") then
        return {}
      end
    end,

    Str = function(el)
      if not el.text:find("„", 1, true) then
        return nil
      end
      local result = pandoc.Inlines({})
      local rest = el.text
      while true do
        local s, e = rest:find("„", 1, true)
        if not s then
          break
        end
        if s > 1 then
          result:insert(pandoc.Str(rest:sub(1, s - 1)))
        end
        result:insert(pandoc.RawInline("typst", '"'))
        rest = rest:sub(e + 1)
      end
      if #rest > 0 then
        result:insert(pandoc.Str(rest))
      end
      return result
    end,

    Math = function(el)
      el.text = el.text
        :gsub("\\left%.", "")
        :gsub("\\right%.", "")
        :gsub("\\left([^%a])", "%1")
        :gsub("\\right([^%a])", "%1")
      return el
    end,

    HorizontalRule = function()
      return {}
    end,

    Div = function(el)
      if el.classes:includes("notes") then
        return el.content
      end
      if el.classes:includes("quelle") then
        local blocks = pandoc.Blocks({ pandoc.RawBlock("typst", "#block(above: 0.4em)[#set text(size: 0.75em, fill: luma(110))") })
        blocks:extend(el.content)
        blocks:insert(pandoc.RawBlock("typst", "]"))
        return blocks
      end
    end,
  },
}
