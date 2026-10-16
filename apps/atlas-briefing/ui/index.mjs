import { useState, useEffect, useMemo, useRef } from 'react'
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from 'react/jsx-runtime'
import { useAppApi, useAppInfo } from '@kirocrew/app-sdk'

const CARD = {
  background: 'var(--card, var(--bg))',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  padding: '14px',
  marginBottom: '12px',
}
const SECTION_TITLE = { fontSize: '13px', fontWeight: 600, color: 'var(--accent, #7c3aed)', marginBottom: '8px' }
const BADGE = (bg, fg) => ({ background: bg, color: fg, padding: '2px 8px', borderRadius: '9999px', fontSize: '10px', fontWeight: 600 })
const MUTED = { fontSize: '11px', color: 'var(--muted)' }

function Icon() {
  return _jsx('svg', {
    xmlns: 'http://www.w3.org/2000/svg', width: 20, height: 20, viewBox: '0 0 24 24',
    fill: 'none', stroke: 'var(--accent, #7c3aed)', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round',
    children: [
      _jsx('circle', { cx: 12, cy: 12, r: 8, key: 'outer' }),
      _jsx('line', { x1: 4, y1: 12, x2: 20, y2: 12, key: 'equator' }),
      _jsx('ellipse', { cx: 12, cy: 12, rx: 3.2, ry: 8, key: 'meridian' }),
    ],
  })
}

function Badge({ children, bg, fg }) {
  return _jsx('span', { style: BADGE(bg || '#e8d5f5', fg || '#7c3aed'), children })
}

function AboutHeader({ overview }) {
  const description = overview?.description
    || 'A read-only dashboard over the Atlas Morning Briefing pipeline.'
  const recent = overview?.recent_briefings || []
  return _jsxs('div', {
    style: { marginBottom: '14px', padding: '0 2px' },
    children: [
      _jsx('div', { style: { fontSize: '12px', lineHeight: 1.5, color: 'var(--text)' }, children: description }),
      recent.length > 0 && _jsxs('div', {
        style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginTop: '10px' },
        children: [
          _jsx('span', { style: { ...MUTED, fontWeight: 500 }, children: 'Recent briefings:' }),
          recent.map(b => _jsx('a', {
            key: b.slug,
            href: b.url,
            style: {
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '3px 10px', borderRadius: '9999px', fontSize: '11px',
              border: '1px solid var(--border)', color: 'var(--accent, #7c3aed)',
              textDecoration: 'none',
            },
            children: `📄 ${b.date}`,
          })),
        ],
      }),
    ],
  })
}

function PipelinePerformance({ overview, loading }) {
  const status = overview?.status || {}
  const errors = status.errors || []
  const ok = errors.length === 0 && (status.papers_found ?? 0) > 0
  return _jsxs('div', {
    style: CARD,
    children: [
      _jsx('div', { style: SECTION_TITLE, children: 'Pipeline Performance' }),
      loading
        ? _jsx('div', { style: MUTED, children: 'Loading…' })
        : _jsxs(_Fragment, {
            children: [
              _jsxs('div', { style: { display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '8px' }, children: [
                _jsx(Stat, { label: 'Papers found', value: status.papers_found ?? '—' }),
                _jsx(Stat, { label: 'News found', value: status.news_found ?? '—' }),
                _jsx(Stat, { label: 'Blogs found', value: status.blogs_found ?? '—' }),
                _jsx(Stat, { label: 'Stocks fetched', value: status.stocks_fetched ?? '—' }),
                _jsx(Stat, { label: 'Topics configured', value: overview?.topic_count ?? '—' }),
              ]}),
              _jsxs('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' }, children: [
                _jsx(Badge, { bg: ok ? '#d1fae5' : '#fef3c7', fg: ok ? '#047857' : '#b45309', children: ok ? '✓ last run clean' : '⚠ last run had issues' }),
                errors.length > 0 && _jsx('span', { style: MUTED, children: errors.join('; ') }),
              ]}),
            ],
          }),
    ],
  })
}

function Stat({ label, value }) {
  return _jsxs('div', { children: [
    _jsx('div', { style: { fontSize: '18px', fontWeight: 600 }, children: String(value) }),
    _jsx('div', { style: MUTED, children: label }),
  ]})
}

function HotTopics({ weekly, loading }) {
  const topics = weekly?.hot_topics || []
  const max = Math.max(1, ...topics.map(t => t.count))
  const [zoom, setZoom] = useState(1)
  const [selected, setSelected] = useState(null)

  const PALETTE = ['#7c3aed', '#2563eb', '#0891b2', '#7c3aed', '#9333ea', '#4f46e5']

  const W = 460, H = 300
  const positions = useMemo(() => {
    const cx = W / 2, cy = H / 2
    const sized = topics.map(t => ({ ...t, r: 22 + (t.count / max) * 34 }))
    const placed = []
    sized.forEach((t, i) => {
      if (i === 0) { placed.push({ ...t, x: cx, y: cy }); return }
      let angle = i * 1.1, radius = t.r + 4, found = null
      for (let step = 0; step < 3000 && !found; step++) {
        const x = cx + radius * Math.cos(angle)
        const y = cy + radius * Math.sin(angle)
        const ok = placed.every(p => {
          const dx = p.x - x, dy = p.y - y
          return Math.sqrt(dx * dx + dy * dy) > p.r + t.r + 2
        })
        if (ok && x - t.r > 2 && x + t.r < W - 2 && y - t.r > 2 && y + t.r < H - 2) {
          found = { x, y }
        }
        angle += 0.3
        radius += 0.8
      }
      placed.push({ ...t, x: found ? found.x : cx, y: found ? found.y : cy })
    })
    return placed
  }, [topics, max])

  // Zoom via viewBox around the center (same idea as the knowledge graph).
  const vbW = W / zoom, vbH = H / zoom
  const vbX = (W - vbW) / 2, vbY = (H - vbH) / 2

  const selectedTopic = topics.find(t => t.term === selected) || null
  const selColor = selectedTopic
    ? PALETTE[positions.findIndex(p => p.term === selected) % PALETTE.length] : null

  return _jsxs('div', {
    style: CARD,
    children: [
      _jsxs('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap' }, children: [
        _jsx('div', { style: SECTION_TITLE, children: `Hot Topics (last ${weekly?.hot_topics_window_days ?? 30} days)` }),
        topics.length > 0 && _jsxs('div', { style: { display: 'flex', gap: '4px' }, children: [
          _jsx('button', { onClick: () => setZoom(z => Math.min(4, z * 1.3)), style: zoomBtn, children: '+' }),
          _jsx('button', { onClick: () => setZoom(z => Math.max(1, z / 1.3)), style: zoomBtn, children: '−' }),
          _jsx('button', { onClick: () => { setZoom(1); setSelected(null) }, style: { ...zoomBtn, width: 'auto', padding: '0 8px' }, children: 'Reset' }),
        ]}),
      ]}),
      loading
        ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'Loading…' })
        : topics.length === 0
          ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'No archived briefings in this window yet.' })
          : _jsxs('div', { style: { display: 'flex', gap: '10px', alignItems: 'stretch', marginTop: '6px' }, children: [
              _jsx('div', { style: { flex: selectedTopic ? '1 1 62%' : '1 1 100%', minWidth: 0 }, children:
                _jsx('svg', {
                  width: '100%', height: 300, viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
                  style: { touchAction: 'none' },
                  onWheel: (evt) => { evt.preventDefault(); setZoom(z => Math.max(1, Math.min(4, z * (evt.deltaY < 0 ? 1.12 : 0.89)))) },
                  children:
                    positions.map((t, i) => {
                      const color = PALETTE[i % PALETTE.length]
                      const byRadius = t.r / 2.2
                      const byLength = (t.r * 1.7) / Math.max(4, t.term.length)
                      const fontSize = Math.max(7, Math.min(13, byRadius, byLength))
                      const isSel = selected === t.term
                      const dim = selected && !isSel
                      return _jsxs('g', {
                        key: t.term, style: { cursor: 'pointer' },
                        onClick: () => setSelected(isSel ? null : t.term),
                        children: [
                          _jsx('circle', {
                            cx: t.x, cy: t.y, r: t.r,
                            fill: color, fillOpacity: dim ? 0.3 : 0.8,
                            stroke: isSel ? 'var(--text)' : color, strokeWidth: isSel ? 2.5 : 1,
                          }),
                          _jsx('text', {
                            x: t.x, y: t.y - 1, textAnchor: 'middle',
                            fontSize, fill: '#fff', fontWeight: 600, style: { pointerEvents: 'none' },
                            children: t.term,
                          }),
                          _jsx('text', {
                            x: t.x, y: t.y + fontSize + 1, textAnchor: 'middle',
                            fontSize: Math.max(7, fontSize - 2), fill: '#fff', fillOpacity: 0.85,
                            style: { pointerEvents: 'none' },
                            children: t.count,
                          }),
                        ],
                      })
                    })
                }),
              }),
              selectedTopic && _jsxs('div', {
                style: { flex: '0 0 36%', borderLeft: '1px solid var(--border)', paddingLeft: '12px', fontSize: '11px', overflowY: 'auto', maxHeight: '300px' },
                children: [
                  _jsxs('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px', marginBottom: '6px' }, children: [
                    _jsxs('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--text)' }, children: [
                      _jsx('span', { style: { width: '10px', height: '10px', borderRadius: '50%', background: selColor, display: 'inline-block' } }),
                      selectedTopic.term,
                    ]}),
                    _jsx('button', { onClick: () => setSelected(null), style: { ...zoomBtn, width: 'auto', padding: '0 6px' }, children: '✕' }),
                  ]}),
                  _jsxs('div', { style: { ...MUTED, marginBottom: '2px' }, children: ['Mentioned ', selectedTopic.count, ' time(s)'] }),
                  (selectedTopic.days || []).length > 0 && _jsxs('div', { style: { ...MUTED, marginBottom: '6px' }, children: ['On ', (selectedTopic.days || []).length, ' day(s): ', (selectedTopic.days || []).join(', ')] }),
                  (selectedTopic.titles || []).length > 0 && _jsxs(_Fragment, { children: [
                    _jsx('div', { style: { fontWeight: 600, color: 'var(--text)', margin: '4px 0 2px' }, children: 'Appears in:' }),
                    _jsx('ul', { style: { margin: 0, paddingLeft: '16px', color: 'var(--text)' }, children:
                      selectedTopic.titles.map((t, i) => _jsx('li', { style: { marginBottom: '3px', lineHeight: 1.35 }, children: t }, i))
                    }),
                  ]}),
                ],
              }),
            ]}),
      _jsx('div', { style: { ...MUTED, marginTop: '6px' }, children: 'Bubble size = frequency across recent papers and news. Scroll or ± to zoom; click a bubble for its details.' }),
    ],
  })
}

function WeeklySummary({ weekly, loading }) {
  const raw = weekly?.summary || ''
  // Parse the LLM's "### Header\nbody" structure into styled sections. Falls
  // back to plain paragraphs when there are no headers (synthesized summary).
  const blocks = useMemo(() => {
    if (!raw) return []
    const out = []
    let current = null
    raw.split('\n').forEach(line => {
      const h = line.match(/^#{2,4}\s+(.+)$/)
      if (h) {
        current = { header: h[1].trim(), body: [] }
        out.push(current)
      } else if (line.trim()) {
        if (!current) { current = { header: null, body: [] }; out.push(current) }
        current.body.push(line.trim())
      }
    })
    return out
  }, [raw])

  return _jsxs('div', {
    style: CARD,
    children: [
      _jsxs('div', { style: { display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }, children: [
        _jsx('div', { style: SECTION_TITLE, children: 'Weekly Summary' }),
        weekly?.summary_meta && _jsx('span', { style: MUTED, children: weekly.summary_meta }),
        weekly?.summary_model && _jsx('span', { style: { ...MUTED, fontStyle: 'italic' }, children: `· ${weekly.summary_model}` }),
      ]}),
      loading
        ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'Loading…' })
        : blocks.length === 0
          ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'No archived editions found in the last 7 days.' })
          : _jsx('div', { style: { marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '10px' }, children:
              blocks.map((blk, i) => _jsxs('div', { key: i, children: [
                blk.header && _jsx('div', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--accent, #7c3aed)', marginBottom: '2px' }, children: blk.header }),
                _jsx('div', { style: { fontSize: '12px', lineHeight: 1.55, color: 'var(--text)' }, children: blk.body.join(' ') }),
              ]}))
            }),
    ],
  })
}

function ResearchTopics({ research, loading }) {
  const topics = research?.topics || []
  const [openIdx, setOpenIdx] = useState(null)
  return _jsxs('div', {
    style: CARD,
    children: [
      _jsxs('div', { style: { ...SECTION_TITLE, display: 'flex', alignItems: 'center', gap: '8px' }, children: [
        _jsx('span', { children: 'Deep-Dive Research Topics' }),
        research && _jsx(Badge, { bg: '#d1fae5', fg: '#047857', children: `${research.validated}/${research.total} validated` }),
      ]}),
      loading
        ? _jsx('div', { style: MUTED, children: 'Loading…' })
        : topics.length === 0
          ? _jsx('div', { style: MUTED, children: 'research-papers repo not found or empty.' })
          : _jsxs(_Fragment, { children: [
              _jsx('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' }, children:
                topics.map((t, i) => {
                  const open = openIdx === i
                  return _jsxs('div', {
                    key: t.name + i,
                    style: { border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden' },
                    children: [
                      _jsxs('div', {
                        style: { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 10px' },
                        children: [
                          _jsx('span', {
                            style: { fontSize: '13px', fontWeight: 500, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                            children: t.name,
                          }),
                          _jsx(Badge, {
                            bg: t.validated ? '#d1fae5' : '#fef3c7',
                            fg: t.validated ? '#047857' : '#b45309',
                            children: t.validated ? '✓ validated' : '◦ in progress',
                          }),
                          t.last_worked && _jsx('span', { style: { ...MUTED, whiteSpace: 'nowrap' }, children: t.last_worked }),
                          _jsx('button', {
                            onClick: () => setOpenIdx(open ? null : i),
                            style: {
                              background: 'transparent', color: 'var(--accent, #7c3aed)',
                              border: '1px solid var(--border)', padding: '3px 10px',
                              borderRadius: '9999px', fontSize: '10px', fontWeight: 500,
                              cursor: 'pointer', whiteSpace: 'nowrap',
                            },
                            children: open ? 'Hide' : 'Details',
                          }),
                        ],
                      }),
                      open && _jsxs('div', {
                        style: { padding: '0 10px 10px 10px', fontSize: '11px', color: 'var(--muted)', borderTop: '1px solid var(--border)', paddingTop: '8px' },
                        children: [
                          t.summary && _jsxs('div', { style: { marginBottom: '6px' }, children: [
                            _jsx('strong', { style: { color: 'var(--text)' }, children: 'Summary: ' }),
                            _jsx('span', { style: { color: 'var(--text)' }, children: t.summary }),
                          ]}),
                          _jsxs('div', { style: { marginBottom: '4px' }, children: [
                            _jsx('strong', { style: { color: 'var(--text)' }, children: 'Source: ' }),
                            `${research?.repo || 'research-papers'}/${t.dir || t.name}`,
                          ]}),
                          _jsxs('div', { style: { marginBottom: '4px' }, children: [
                            _jsx('strong', { style: { color: 'var(--text)' }, children: 'Evaluation: ' }),
                            t.validated
                              ? `Validated — has a recorded paper draft, findings, or results (${t.signal || 'artifact found'}).`
                              : 'In progress — no paper draft, findings, or results artifact yet.',
                          ]}),
                          t.last_worked && _jsxs('div', { style: { marginBottom: '4px' }, children: [
                            _jsx('strong', { style: { color: 'var(--text)' }, children: 'Last worked: ' }),
                            t.last_worked,
                          ]}),
                          typeof t.activity === 'number' && _jsxs('div', { children: [
                            _jsx('strong', { style: { color: 'var(--text)' }, children: 'Activity: ' }),
                            `${t.activity} markdown/code files`,
                          ]}),
                        ],
                      }),
                    ],
                  })
                })
              }),
              _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children:
                `Top ${topics.length} of ${research?.total_available ?? topics.length} deep-dives from ${research?.repo || 'research-papers'}, sorted by most recently worked on.` }),
            ]}),
    ],
  })
}

function useLiveForceLayout(nodes, edges, width, height) {
  // A genuinely LIVE force simulation (not a one-shot static layout): runs
  // continuously via requestAnimationFrame so the graph keeps settling and
  // responds to drag. External apps cannot import the host's own graph
  // engine (sigma/graphology/d3 are not in the shared-module allowlist handed
  // to federated app bundles -- only react/react-dom/jsx-runtime/lucide-react/
  // react-query/@kirocrew/app-sdk/@kirocrew/ui are), so this hand-rolled
  // simulation is the closest an external app can get to that interactivity.
  const posRef = useRef({})
  const velRef = useRef({})
  const draggingRef = useRef(null)
  const [, forceRender] = useState(0)

  const nodeIds = nodes.map(n => n.id).join(',')
  const edgeKey = edges.map(e => `${e.source}-${e.target}`).join(',')

  useEffect(() => {
    const cx = width / 2, cy = height / 2
    const pos = posRef.current, vel = velRef.current
    // Keep existing positions for nodes that persist across a data refresh;
    // seed new ones on a ring so they ease in rather than popping to center.
    const liveIds = new Set(nodes.map(n => n.id))
    Object.keys(pos).forEach(id => { if (!liveIds.has(id)) { delete pos[id]; delete vel[id] } })
    nodes.forEach((node, i) => {
      if (!pos[node.id]) {
        const angle = (i / Math.max(1, nodes.length)) * 2 * Math.PI
        const r = Math.min(width, height) * 0.32
        pos[node.id] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }
        vel[node.id] = { x: 0, y: 0 }
      }
    })

    const edgeSet = edges.map(e => [e.source, e.target, e.weight || 1])
    const REPEL = 2600, SPRING = 0.02, IDEAL_LEN = 70, DAMPING = 0.82
    let raf = null
    let settleFrames = 0

    function tick() {
      const n = nodes.length
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = nodes[i].id, b = nodes[j].id
          if (!pos[a] || !pos[b]) continue
          let dx = pos[a].x - pos[b].x, dy = pos[a].y - pos[b].y
          let distSq = dx * dx + dy * dy
          if (distSq < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; distSq = 1 }
          const dist = Math.sqrt(distSq)
          const force = REPEL / distSq
          const fx = (dx / dist) * force, fy = (dy / dist) * force
          if (draggingRef.current !== a) { vel[a].x += fx; vel[a].y += fy }
          if (draggingRef.current !== b) { vel[b].x -= fx; vel[b].y -= fy }
        }
      }
      edgeSet.forEach(([a, b, w]) => {
        if (!pos[a] || !pos[b]) return
        const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y
        const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
        const force = SPRING * (dist - IDEAL_LEN)
        const fx = (dx / dist) * force, fy = (dy / dist) * force
        if (draggingRef.current !== a) { vel[a].x += fx; vel[a].y += fy }
        if (draggingRef.current !== b) { vel[b].x -= fx; vel[b].y -= fy }
      })
      let maxSpeed = 0
      nodes.forEach(node => {
        const p = pos[node.id], v = vel[node.id]
        if (draggingRef.current === node.id) return // pinned while dragged
        v.x += (cx - p.x) * 0.0015
        v.y += (cy - p.y) * 0.0015
        v.x *= DAMPING; v.y *= DAMPING
        p.x += v.x; p.y += v.y
        const pad = 26
        p.x = Math.max(pad, Math.min(width - pad, p.x))
        p.y = Math.max(pad, Math.min(height - pad, p.y))
        maxSpeed = Math.max(maxSpeed, Math.abs(v.x) + Math.abs(v.y))
      })
      forceRender(x => x + 1)
      // Keep animating while anything is still moving meaningfully, or while
      // a node is being dragged; stop burning frames once fully settled.
      if (maxSpeed > 0.05 || draggingRef.current) {
        settleFrames = 0
        raf = requestAnimationFrame(tick)
      } else if (settleFrames < 3) {
        settleFrames += 1
        raf = requestAnimationFrame(tick)
      }
    }
    raf = requestAnimationFrame(tick)
    return () => { if (raf) cancelAnimationFrame(raf) }
  }, [nodeIds, edgeKey, width, height])

  function startDrag(id) {
    draggingRef.current = id
    // Wake the simulation again once a drag starts after it had settled.
    forceRender(x => x + 1)
  }
  function dragTo(id, x, y) {
    if (posRef.current[id]) { posRef.current[id].x = x; posRef.current[id].y = y }
  }
  function endDrag() {
    draggingRef.current = null
    forceRender(x => x + 1)
  }

  return { positions: posRef.current, startDrag, dragTo, endDrag, draggingId: draggingRef.current }
}

function KnowledgeGraph({ graph, loading }) {
  const { nodes, edges } = graph || { nodes: [], edges: [] }
  const W = 640, H = 380
  const svgRef = useRef(null)
  const { positions, startDrag, dragTo, endDrag } = useLiveForceLayout(nodes, edges, W, H)
  const [hoverId, setHoverId] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [zoom, setZoom] = useState(1)       // 1 = fit; >1 = zoomed in
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const maxWeight = Math.max(1, ...nodes.map(n => n.weight || 1))

  const CATEGORY_COLOR = {
    service: '#3b82f6',
    technology: '#22c55e',
    concept: '#a855f7',
    org: '#f97316',
  }
  const categories = graph?.categories?.length ? graph.categories : Object.keys(CATEGORY_COLOR)

  // Zoom/pan drives the SVG viewBox: zooming in shrinks the viewBox around the
  // current center, which is how the native graph reveals finer labels as you
  // zoom in. Clamped so you cannot lose the graph off-screen.
  const vbW = W / zoom, vbH = H / zoom
  const vbX = Math.max(-W, Math.min(W, (W - vbW) / 2 + pan.x))
  const vbY = Math.max(-H, Math.min(H, (H - vbH) / 2 + pan.y))

  function svgPoint(evt) {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const rect = svg.getBoundingClientRect()
    const x = vbX + ((evt.clientX - rect.left) / rect.width) * vbW
    const y = vbY + ((evt.clientY - rect.top) / rect.height) * vbH
    return { x, y }
  }

  const neighborIds = useMemo(() => {
    const anchor = selectedId || hoverId
    if (!anchor) return null
    const s = new Set([anchor])
    edges.forEach(e => {
      if (e.source === anchor) s.add(e.target)
      if (e.target === anchor) s.add(e.source)
    })
    return s
  }, [hoverId, selectedId, edges])

  const selectedNode = nodes.find(n => n.id === selectedId) || null
  const selColor = selectedNode ? (CATEGORY_COLOR[selectedNode.category] || '#94a3b8') : null

  // Default to showing every node's label -- a sparse 18-24 node graph is
  // exactly where native graph views keep labels on by default; thresholding
  // by weight (the previous behavior) hid all but the single heaviest node
  // when most weights tie at 1. Zoom still sharpens/enlarges text via the
  // font-size calc below, it no longer gates whether a label exists at all.
  function labelVisible(n, w, dim) {
    if (dim) return false
    return true
  }

  return _jsxs('div', {
    style: CARD,
    children: [
      _jsxs('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '8px' }, children: [
        _jsx('div', { style: SECTION_TITLE, children: `Knowledge Graph (topic co-occurrence, last ${graph?.window_days ?? 30} days)` }),
        _jsx('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' }, children:
          categories.map(c => _jsxs('span', {
            key: c,
            style: { display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--muted)' },
            children: [
              _jsx('span', { style: { width: '8px', height: '8px', borderRadius: '50%', background: CATEGORY_COLOR[c] || '#94a3b8', display: 'inline-block' } }),
              c,
            ],
          }))
        }),
      ]}),
      loading
        ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'Loading…' })
        : nodes.length === 0
          ? _jsx('div', { style: { ...MUTED, marginTop: '8px' }, children: 'Nothing to graph yet — needs at least one archived briefing.' })
          : _jsxs(_Fragment, { children: [
              _jsxs('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '8px 0 4px', gap: '8px' }, children: [
                _jsx('div', { style: MUTED, children: `${nodes.length} nodes, ${edges.length} edges — drag a node, scroll/±  to zoom, click to inspect` }),
                _jsxs('div', { style: { display: 'flex', gap: '4px' }, children: [
                  _jsx('button', { onClick: () => setZoom(z => Math.min(4, z * 1.3)), style: zoomBtn, children: '+' }),
                  _jsx('button', { onClick: () => setZoom(z => Math.max(1, z / 1.3)), style: zoomBtn, children: '−' }),
                  _jsx('button', { onClick: () => { setZoom(1); setPan({ x: 0, y: 0 }) }, style: { ...zoomBtn, width: 'auto', padding: '0 8px' }, children: 'Recenter' }),
                ]}),
              ]}),
              _jsxs('div', { style: { display: 'flex', gap: '10px', alignItems: 'stretch' }, children: [
                _jsx('div', { style: { flex: selectedNode ? '1 1 62%' : '1 1 100%', minWidth: 0 }, children:
                  _jsxs('svg', {
                    ref: svgRef, width: '100%', height: 380, viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
                    style: { cursor: 'default', touchAction: 'none' },
                    onWheel: (evt) => { evt.preventDefault(); setZoom(z => Math.max(1, Math.min(4, z * (evt.deltaY < 0 ? 1.12 : 0.89)))) },
                    onMouseUp: () => endDrag(),
                    onMouseLeave: () => { endDrag(); setHoverId(null) },
                    children: [
                      edges.map((e, i) => {
                        const a = positions[e.source], b = positions[e.target]
                        if (!a || !b) return null
                        const dim = neighborIds && !(neighborIds.has(e.source) && neighborIds.has(e.target))
                        return _jsx('line', {
                          key: `e${i}`, x1: a.x, y1: a.y, x2: b.x, y2: b.y,
                          stroke: 'var(--border)', strokeWidth: Math.min(3, 0.5 + e.weight * 0.5),
                          opacity: dim ? 0.12 : 0.7,
                        })
                      }),
                      nodes.map(n => {
                        const p = positions[n.id]
                        if (!p) return null
                        const w = n.weight || 1
                        const radius = 5 + Math.min(16, (w / maxWeight) * 18)
                        const color = CATEGORY_COLOR[n.category] || '#94a3b8'
                        const dim = neighborIds && !neighborIds.has(n.id)
                        const isSel = selectedId === n.id
                        return _jsxs('g', {
                          key: n.id,
                          style: { cursor: 'pointer' },
                          onMouseDown: (evt) => { evt.preventDefault(); startDrag(n.id) },
                          onMouseMove: (evt) => { if (evt.buttons === 1) { const pt = svgPoint(evt); dragTo(n.id, pt.x, pt.y) } },
                          onMouseEnter: () => setHoverId(n.id),
                          onMouseLeave: () => setHoverId(null),
                          onClick: () => setSelectedId(isSel ? null : n.id),
                          children: [
                            _jsx('circle', {
                              cx: p.x, cy: p.y, r: radius,
                              fill: color, fillOpacity: dim ? 0.25 : 0.8,
                              stroke: isSel ? 'var(--text)' : color, strokeWidth: isSel ? 2.5 : (hoverId === n.id ? 2 : 1),
                            }),
                            labelVisible(n, w, dim) && _jsx('text', {
                              x: p.x, y: p.y + radius + 11, textAnchor: 'middle',
                              fontSize: Math.max(7, 10 / Math.sqrt(zoom)), fill: 'var(--text)',
                              style: { pointerEvents: 'none' },
                              children: n.label,
                            }),
                          ],
                        })
                      }),
                    ],
                  }),
                }),
                // Right-side detail panel, shown when a node is clicked (as the
                // native KnowledgeGraph does).
                selectedNode && _jsxs('div', {
                  style: { flex: '0 0 36%', borderLeft: '1px solid var(--border)', paddingLeft: '12px', fontSize: '11px', overflowY: 'auto', maxHeight: '380px' },
                  children: [
                    _jsxs('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px', marginBottom: '6px' }, children: [
                      _jsxs('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: 'var(--text)' }, children: [
                        _jsx('span', { style: { width: '10px', height: '10px', borderRadius: '50%', background: selColor, display: 'inline-block' } }),
                        selectedNode.label,
                      ]}),
                      _jsx('button', { onClick: () => setSelectedId(null), style: { ...zoomBtn, width: 'auto', padding: '0 6px' }, children: '✕' }),
                    ]}),
                    _jsxs('div', { style: { ...MUTED, marginBottom: '2px' }, children: ['Category: ', selectedNode.category] }),
                    _jsxs('div', { style: { ...MUTED, marginBottom: '2px' }, children: ['Mentioned on ', (selectedNode.days || []).length, ' day(s): ', (selectedNode.days || []).join(', ')] }),
                    _jsxs('div', { style: { ...MUTED, marginBottom: '6px' }, children: ['Connected to ', neighborIds ? neighborIds.size - 1 : 0, ' topic(s)'] }),
                    (selectedNode.titles || []).length > 0 && _jsxs(_Fragment, { children: [
                      _jsx('div', { style: { fontWeight: 600, color: 'var(--text)', margin: '4px 0 2px' }, children: 'Appears in:' }),
                      _jsx('ul', { style: { margin: 0, paddingLeft: '16px', color: 'var(--text)' }, children:
                        selectedNode.titles.map((t, i) => _jsx('li', { style: { marginBottom: '3px', lineHeight: 1.35 }, children: t }, i))
                      }),
                    ]}),
                  ],
                }),
              ]}),
            ]}),
      _jsx('div', { style: { ...MUTED, marginTop: '6px' }, children: 'Topic phrases from recent Top Papers and News. Live physics layout; drag nodes, scroll or ± to zoom (labels sharpen as you zoom in), click a node for its details on the right. Node size = importance; color = category.' }),
    ],
  })
}

const zoomBtn = {
  width: '24px', height: '22px', borderRadius: '6px', border: '1px solid var(--border)',
  background: 'transparent', color: 'var(--text)', cursor: 'pointer', fontSize: '13px', lineHeight: 1,
}

export default function AtlasBriefingApp() {
  const api = useAppApi()
  const { version: appVersion } = useAppInfo()
  const [overview, setOverview] = useState(null)
  const [weekly, setWeekly] = useState(null)
  const [graph, setGraph] = useState(null)
  const [research, setResearch] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)

  async function loadAll() {
    try {
      const [ov, wk, kg, rt] = await Promise.all([
        api.get('/api/apps/atlas-briefing/overview'),
        api.get('/api/apps/atlas-briefing/weekly'),
        api.get('/api/apps/atlas-briefing/knowledge-graph'),
        api.get('/api/apps/atlas-briefing/research-topics'),
      ])
      setOverview(ov)
      setWeekly(wk)
      setGraph(kg)
      setResearch(rt)
      setLastRefresh(new Date())
    } catch (e) {
      // Keep whatever we last had; show stale data rather than a blank page.
      console.error('atlas-briefing load failed', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAll()
    const interval = setInterval(loadAll, 30000)
    return () => clearInterval(interval)
  }, [])

  return _jsxs('div', {
    style: { maxWidth: '1200px', margin: '0 auto', padding: '16px' },
    children: [
      _jsxs('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' },
        children: [
          _jsxs('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' }, children: [
            _jsx(Icon, {}),
            _jsx('h2', { style: { margin: 0, fontSize: '18px' }, children: 'Atlas Briefing' }),
            _jsx(Badge, { children: 'Daily 08:00 CEST' }),
          ]}),
          _jsxs('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' }, children: [
            _jsx('span', { style: MUTED, children: lastRefresh ? `Refreshed ${lastRefresh.toLocaleTimeString()}` : '' }),
            _jsx('button', {
              disabled: loading,
              onClick: () => { setLoading(true); loadAll() },
              style: { background: 'transparent', color: loading ? 'var(--muted)' : 'var(--accent, #7c3aed)', border: '1px solid var(--border)', padding: '5px 14px', borderRadius: '9999px', fontSize: '11px', fontWeight: 500, cursor: loading ? 'default' : 'pointer' },
              children: loading ? '↻ Loading…' : '↻ Refresh',
            }),
            _jsx('span', { style: { fontSize: '10px', color: 'var(--muted)' }, children: `v${appVersion || '1.0.0'}` }),
          ]}),
        ],
      }),

      _jsx(AboutHeader, { overview }),

      _jsx(PipelinePerformance, { overview, loading }),

      _jsx(WeeklySummary, { weekly, loading }),

      _jsx(ResearchTopics, { research, loading }),

      _jsx(HotTopics, { weekly, loading }),

      _jsx(KnowledgeGraph, { graph, loading }),
    ],
  })
}
