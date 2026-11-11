locals {
  services = [
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
  ]
  db_name = "niyam"
  db_user = "niyam"
  socket  = "/cloudsql/${google_sql_database_instance.main.connection_name}"
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "niyam"
  location      = var.region
  format        = "DOCKER"
  description   = "Niyam container images"
  depends_on    = [google_project_service.enabled]
}

# ---------------------------------------------------------------- database

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "main" {
  name                = "niyam-pg"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.db_tier
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"
    disk_size         = 10
    disk_autoresize   = true

    ip_configuration {
      # Public IP with no authorised networks: only the Cloud SQL connector (IAM-checked) can reach it.
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    insights_config {
      query_insights_enabled = true
    }
  }

  depends_on = [google_project_service.enabled]
}

resource "google_sql_database" "niyam" {
  name     = local.db_name
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "niyam" {
  name     = local.db_user
  instance = google_sql_database_instance.main.name
  password = random_password.db.result
}

# ---------------------------------------------------------------- secrets

resource "google_secret_manager_secret" "database_url" {
  secret_id = "niyam-database-url"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${local.db_user}:${random_password.db.result}@/${local.db_name}?host=${local.socket}"
}

# The value is added out of band so the key never lands in Terraform state:
#   printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets versions add niyam-anthropic-api-key --data-file=-
resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "niyam-anthropic-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}

# ---------------------------------------------------------------- identities

resource "google_service_account" "api" {
  account_id   = "niyam-api"
  display_name = "Niyam API and ingestion"
}

resource "google_service_account" "scheduler" {
  account_id   = "niyam-scheduler"
  display_name = "Niyam ingestion scheduler"
}

resource "google_project_iam_member" "api_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "api_secrets" {
  for_each = {
    database_url      = google_secret_manager_secret.database_url.id
    anthropic_api_key = google_secret_manager_secret.anthropic_api_key.id
  }
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# ---------------------------------------------------------------- API service

resource "google_cloud_run_v2_service" "api" {
  name                = "niyam-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.api.email
    max_instance_request_concurrency = 20
    timeout                          = "120s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.api_max_instances
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.main.connection_name]
      }
    }

    containers {
      image = var.api_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "NIYAM_ENV"
        value = "prod"
      }
      env {
        name  = "NIYAM_LOG_JSON"
        value = "true"
      }
      env {
        name  = "NIYAM_CORS_ORIGINS"
        value = jsonencode(var.cors_origins)
      }
      env {
        name = "NIYAM_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_api_key.secret_id
            version = "latest"
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 24
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [
    google_project_iam_member.api_sql,
    google_secret_manager_secret_iam_member.api_secrets,
    google_secret_manager_secret_version.database_url,
  ]
}

# ---------------------------------------------------------------- web service

resource "google_cloud_run_v2_service" "web" {
  name                = "niyam-web"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = var.web_image
      ports {
        container_port = 3000
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }
    }
  }

  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  for_each = var.public ? {
    api = google_cloud_run_v2_service.api.name
    web = google_cloud_run_v2_service.web.name
  } : {}
  name     = each.value
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------- daily ingestion

resource "google_cloud_run_v2_job" "ingest" {
  name                = "niyam-ingest"
  location            = var.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.api.email
      timeout         = "3600s"
      max_retries     = 1

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.main.connection_name]
        }
      }

      containers {
        image   = var.api_image
        command = ["sh", "-c"]
        args    = ["niyam ingest rbi && niyam embed"]

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        env {
          name  = "NIYAM_LOG_JSON"
          value = "true"
        }
        env {
          name = "NIYAM_DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.api_sql,
    google_secret_manager_secret_iam_member.api_secrets,
    google_secret_manager_secret_version.database_url,
  ]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_runs_ingest" {
  name     = google_cloud_run_v2_job.ingest.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "ingest_daily" {
  name      = "niyam-ingest-daily"
  region    = var.region
  schedule  = var.ingest_schedule
  time_zone = "Asia/Kolkata"

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.ingest.name}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler_runs_ingest]
}
