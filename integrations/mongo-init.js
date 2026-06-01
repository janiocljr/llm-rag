// =============================================================================
// mongo-init.js — RAG Knowledge Base initialisation
// =============================================================================
// Runs once when the MongoDB container is first created.
// Creates the application user, the database, all collections, and indexes.
// =============================================================================

// Switch to the knowledge base database
const db = db.getSiblingDB(process.env.MONGO_INITDB_DATABASE || "rag_knowledge");

// ---------------------------------------------------------------------------
// Application user (least-privilege — readWrite on this DB only)
// ---------------------------------------------------------------------------
db.createUser({
  user: "raguser",
  pwd: "ragpassword",       // override via MONGO_APP_PASSWORD env if needed
  roles: [{ role: "readWrite", db: db.getName() }],
});

// ---------------------------------------------------------------------------
// Collection: documents
// Obsidian-style markdown documents — the human-readable knowledge base.
// ---------------------------------------------------------------------------
db.createCollection("documents", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["title", "content", "doc_type", "created_at"],
      properties: {
        title:        { bsonType: "string",   description: "Document title" },
        content:      { bsonType: "string",   description: "Full markdown content" },
        doc_type:     {
          bsonType: "string",
          enum: ["note", "task", "question", "conversation", "knowledge", "pdf_chunk"],
          description: "Document category",
        },
        tags:         { bsonType: "array",    items: { bsonType: "string" } },
        status:       { bsonType: "string",   enum: ["active", "archived", "draft"] },
        source_file:  { bsonType: "string",   description: "Origin PDF filename if any" },
        page_number:  { bsonType: "int",      description: "Origin page number" },
        session_id:   { bsonType: "string",   description: "Chat session reference" },
        chroma_ids:   { bsonType: "array",    description: "ChromaDB embedding IDs for this document" },
        created_at:   { bsonType: "date" },
        updated_at:   { bsonType: "date" },
        metadata:     { bsonType: "object",   description: "Arbitrary extra metadata" },
      },
    },
  },
});

db.documents.createIndex({ title: "text", content: "text", tags: "text" });
db.documents.createIndex({ doc_type: 1 });
db.documents.createIndex({ tags: 1 });
db.documents.createIndex({ session_id: 1 });
db.documents.createIndex({ source_file: 1 });
db.documents.createIndex({ created_at: -1 });
db.documents.createIndex({ chroma_ids: 1 });

// ---------------------------------------------------------------------------
// Collection: sessions
// One document per chat session — stores the full turn history and metadata.
// ---------------------------------------------------------------------------
db.createCollection("sessions");

db.sessions.createIndex({ started_at: -1 });
db.sessions.createIndex({ tags: 1 });

// ---------------------------------------------------------------------------
// Collection: tasks
// Structured task tracking with status and priority.
// ---------------------------------------------------------------------------
db.createCollection("tasks");

db.tasks.createIndex({ status: 1 });
db.tasks.createIndex({ priority: 1 });
db.tasks.createIndex({ due_date: 1 });
db.tasks.createIndex({ session_id: 1 });
db.tasks.createIndex({ created_at: -1 });

// ---------------------------------------------------------------------------
// Collection: memory_snapshots
// Periodic snapshots of ChromaDB collection stats for observability.
// ---------------------------------------------------------------------------
db.createCollection("memory_snapshots");
db.memory_snapshots.createIndex({ captured_at: -1 });

print("✅ MongoDB initialisation complete — database: " + db.getName());
