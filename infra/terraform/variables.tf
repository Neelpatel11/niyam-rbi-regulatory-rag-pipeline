variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Cloud SQL and Artifact Registry."
  type        = string
  default     = "asia-south1"
}

variable "api_image" {
  description = "Full image reference for the API, e.g. asia-south1-docker.pkg.dev/<project>/niyam/api:<sha>."
  type        = string
}

variable "web_image" {
  description = "Full image reference for the web UI. Build it with NEXT_PUBLIC_API_URL set to the API URL."
  type        = string
}

variable "db_tier" {
  description = "Cloud SQL machine tier."
  type        = string
  default     = "db-custom-1-3840"
}

variable "deletion_protection" {
  description = "Protect the database instance from accidental deletion."
  type        = bool
  default     = true
}

variable "public" {
  description = "Allow unauthenticated access to the API and web services."
  type        = bool
  default     = true
}

variable "api_max_instances" {
  description = "Upper bound on API instances. Keeps LLM spend bounded under load."
  type        = number
  default     = 3
}

variable "ingest_schedule" {
  description = "Cron schedule for the ingestion job (Asia/Kolkata)."
  type        = string
  default     = "30 2 * * *"
}

variable "cors_origins" {
  description = "Origins allowed to call the API from a browser."
  type        = list(string)
  default     = []
}
