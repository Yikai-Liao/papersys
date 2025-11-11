-- Removes non-semantic wrappers so pandoc emits cleaner Markdown.
-- Div/Span are flattened if they only provide styling hooks.

local function has_ar5iv_class(element)
  if not element.classes then
    return false
  end
  for _, class in ipairs(element.classes) do
    if class:match("^ar5iv%-") then
      return true
    end
  end
  return false
end

function Div(elem)
  if has_ar5iv_class(elem) then
    return {}
  end
  return elem.content
end

local function collect_inlines(blocks)
  local inlines = {}
  for _, block in ipairs(blocks or {}) do
    if block.t == "Plain" or block.t == "Para" then
      for _, inline in ipairs(block.content) do
        table.insert(inlines, inline)
      end
    elseif block.t == "LineBlock" then
      for i, line in ipairs(block.content) do
        for _, inline in ipairs(line) do
          table.insert(inlines, inline)
        end
        if i < #block.content then
          table.insert(inlines, pandoc.SoftBreak())
        end
      end
    else
      local text = pandoc.utils.stringify(block)
      if text ~= "" then
        table.insert(inlines, pandoc.Str(text))
      end
    end
  end
  return inlines
end

function Table(elem)
  -- Check if this is an equation table (ltx_equation ltx_eqn_table)
  if elem.classes then
    for _, class in ipairs(elem.classes) do
      if class:match("ltx_equation") or class:match("ltx_eqn") then
        -- Extract the math content and equation number from table cells
        local math_inlines = {}
        local eqno_text = nil
        if elem.bodies and #elem.bodies > 0 then
          for _, body in ipairs(elem.bodies) do
            for _, row in ipairs(body.body) do
              for _, cell in ipairs(row.cells) do
                local is_eqno = false
                if cell.attr and cell.attr.classes then
                  for _, c in ipairs(cell.attr.classes) do
                    if c == "ltx_eqn_eqno" then
                      is_eqno = true
                      break
                    end
                  end
                end
                local inlines = collect_inlines(cell.contents)
                if is_eqno then
                  eqno_text = pandoc.utils.stringify(pandoc.Plain(inlines))
                else
                  for _, inline in ipairs(inlines) do
                    table.insert(math_inlines, inline)
                  end
                end
              end
            end
          end
        end
        if #math_inlines > 0 then
          local math_text = pandoc.utils.stringify(pandoc.Plain(math_inlines))
          local blocks = {
            pandoc.Para({
              pandoc.Math("DisplayMath", math_text)
            })
          }
          if eqno_text and eqno_text ~= "" then
            table.insert(blocks, pandoc.RawBlock("markdown", eqno_text))
          end
          return blocks
        end
        return {}
      end
    end
  end
  return elem
end



function Link(elem)
  if has_ar5iv_class(elem) then
    return {}
  end
  return elem.content
end

local function caption_to_blocks(caption)
  if not caption then
    return {}
  end

  local caption_type = pandoc.utils.type(caption)
  -- Pandoc 3.6+ returns Caption userdata with .long blocks.
  if caption_type == "Caption" then
    return caption.long or {}
  end

  -- Older versions already return block lists or inline lists; forward as-is.
  if type(caption) == "table" then
    return caption
  end

  return {}
end

function Figure(elem)
  local caption_blocks = caption_to_blocks(elem.caption)
  if caption_blocks and #caption_blocks > 0 then
    return caption_blocks
  end
  return {}
end

-- Remove HTML <sup>...</sup> wrappers appearing in RawInline HTML fragments.
-- Collapses adjacent identical <sup> blocks (e.g. <sup>2</sup><sup>2</sup>)
-- then strips remaining <sup>...</sup> keeping their inner text.
function RawInline(elem)
  if elem.format and elem.format:match('html') then
    local s = elem.text
    -- collapse adjacent identical <sup>...</sup> sequences
    local prev = nil
    repeat
      prev = s
      s = s:gsub('<sup>(.-)</sup>%s*<sup>%1</sup>', '<sup>%1</sup>')
    until s == prev
    -- strip remaining <sup> tags but keep their contents
    local stripped = s:gsub('<sup>(.-)</sup>', '%1')
    if stripped ~= elem.text then
      return pandoc.Str(stripped)
    end
  end
  return nil
end

-- Clean up LaTeX math expressions by removing unnecessary \hspace{0pt} commands
function Math(elem)
  local cleaned = elem.text
  -- Remove \hspace{0pt} which is zero-width space (meaningless)
  cleaned = cleaned:gsub('\\hspace%{0pt%}', '')
  -- Remove other common LaTeXML artifacts if needed
  if cleaned ~= elem.text then
    return pandoc.Math(elem.mathtype, cleaned)
  end
  return elem
end

-- Remove LaTeXML attributes from headers ({#id .class1 .class2})
function Header(elem)
  -- Clear identifier and classes that come from LaTeXML
  elem.identifier = ""
  elem.classes = {}
  elem.attributes = {}
  return elem
end

-- Handle nested Span elements that create duplicate footnote markers
-- LaTeXML wraps footnotes in nested spans with multiple superscripts
function Span(elem)
  if has_ar5iv_class(elem) then
    return {}
  end
  
  -- Check for ltx_note class patterns that contain nested duplicate content
  if elem.classes then
    for _, class in ipairs(elem.classes) do
      if class:match("ltx_note") then
        -- For footnote spans, extract only the first superscript marker
        -- This prevents ^1^^1^footnotemark patterns
        local first_sup = nil
        for _, item in ipairs(elem.content) do
          if item.t == "Superscript" and not first_sup then
            first_sup = item
            break
          end
        end
        if first_sup then
          return {first_sup}
        end
      end
      -- Remove bullet point markers from list items (ltx_tag_item)
      -- These create duplicate bullets: - • when Markdown already adds -
      if class:match("ltx_tag_item") then
        return {}
      end
    end
  end
  
  return elem.content
end
