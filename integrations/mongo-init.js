const db = db.getSiblingDB(process.env.MONGO_INITDB_DATABASE || "rag_knowledge");

db.createUser({
  user: "raguser",
  pwd: "ragpassword",
  roles: [{ role: "readWrite", db: db.getName() }],
});

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

db.createCollection("sessions");

db.sessions.createIndex({ started_at: -1 });
db.sessions.createIndex({ tags: 1 });

db.createCollection("tasks");

db.tasks.createIndex({ status: 1 });
db.tasks.createIndex({ priority: 1 });
db.tasks.createIndex({ due_date: 1 });
db.tasks.createIndex({ session_id: 1 });
db.tasks.createIndex({ created_at: -1 });

db.createCollection("memory_snapshots");
db.memory_snapshots.createIndex({ captured_at: -1 });

print("✅ MongoDB initialisation complete — database: " + db.getName());
