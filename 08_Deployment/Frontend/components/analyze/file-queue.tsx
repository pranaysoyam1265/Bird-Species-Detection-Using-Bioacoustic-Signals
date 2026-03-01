"use client"

import { useState, useCallback } from "react"
import { X, Play, Trash2, CheckCircle, Loader2, Clock, AlertCircle } from "lucide-react"

export type QueueItemStatus = "pending" | "processing" | "complete" | "error"

export interface QueueItem {
  file: File
  status: QueueItemStatus
  id: string
}

interface FileQueueProps {
  items: QueueItem[]
  onRemove: (id: string) => void
  onClear: () => void
  onProcess: () => void
  onSelectItem: (item: QueueItem) => void
}

export function FileQueue({ items, onRemove, onClear, onProcess, onSelectItem }: FileQueueProps) {
  if (items.length === 0) return null

  const statusIcon = (status: QueueItemStatus) => {
    switch (status) {
      case "complete": return <CheckCircle size={14} className="text-green-500" />
      case "processing": return <Loader2 size={14} className="text-accent animate-spin" />
      case "error": return <AlertCircle size={14} className="text-red-500" />
      default: return <Clock size={14} className="text-muted-foreground" />
    }
  }

  const statusLabel = (status: QueueItemStatus) => {
    switch (status) {
      case "complete": return "COMPLETE"
      case "processing": return "ANALYZING"
      case "error": return "ERROR"
      default: return "PENDING"
    }
  }

  const hasProcessable = items.some((i) => i.status === "pending")

  return (
    <div className="border-2 border-foreground bg-background">
      <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-2">
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          FILE QUEUE ({items.length} {items.length === 1 ? "file" : "files"})
        </span>
        <button
          type="button"
          onClick={onClear}
          className="font-mono text-[10px] tracking-wider uppercase text-red-500 hover:text-red-400 cursor-pointer border-none bg-transparent p-0"
        >
          CLEAR
        </button>
      </div>
      <div className="divide-y divide-foreground/20">
        {items.map((item) => (
          <div
            key={item.id}
            className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/30 cursor-pointer"
            onClick={() => onSelectItem(item)}
          >
            {statusIcon(item.status)}
            <span className="font-mono text-xs tracking-wider text-foreground truncate flex-1">
              {item.file.name}
            </span>
            <span className="font-mono text-[10px] tracking-wider text-muted-foreground shrink-0">
              {(item.file.size / (1024 * 1024)).toFixed(1)} MB
            </span>
            <span className={`font-mono text-[10px] tracking-wider shrink-0 w-[72px] text-right ${item.status === "complete" ? "text-green-500" :
              item.status === "processing" ? "text-accent" :
                item.status === "error" ? "text-red-500" :
                  "text-muted-foreground"
              }`}>
              {statusLabel(item.status)}
            </span>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onRemove(item.id); }}
              className="text-muted-foreground hover:text-red-500 cursor-pointer"
            >
              <X size={12} />
            </button>
          </div>
        ))}
      </div>
      {hasProcessable && (
        <div className="border-t-2 border-foreground p-3">
          <button
            type="button"
            onClick={onProcess}
            className="w-full py-2 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.15em] uppercase font-bold cursor-pointer transition-none hover:bg-background hover:text-accent flex items-center justify-center gap-2"
          >
            <Play size={12} /> START BATCH ANALYSIS
          </button>
        </div>
      )}
    </div>
  )
}
