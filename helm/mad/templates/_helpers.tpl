{{- define "retrain-scheduler.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Convert retrain interval to cron schedule
*/}}
{{- define "retrain.cronSchedule" -}}
{{- $interval := . -}}
{{- if regexMatch "^[0-9]+d$" $interval -}}
{{- $days := trimSuffix "d" $interval | int -}}
{{- if eq $days 1 }}0 0 * * *{{- end -}}
{{- if eq $days 7 }}0 0 * * 0{{- end -}}
{{- else if regexMatch "^[0-9]+h$" $interval -}}
{{- $hours := trimSuffix "h" $interval | int -}}
{{- if eq $hours 24 }}0 * * * *{{- end -}}
{{- if eq $hours 12 }}0 */12 * * *{{- end -}}
{{- if eq $hours 6 }}0 */6 * * *{{- end -}}
{{- if eq $hours 1 }}0 * * * *{{- end -}}
{{- else if regexMatch "^[0-9]+m$" $interval -}}
{{- $minutes := trimSuffix "m" $interval | int -}}
{{- if eq $minutes 30 }}*/30 * * * *{{- end -}}
{{- if eq $minutes 15 }}*/15 * * * *{{- end -}}
{{- else -}}0 0 * * *{{/* default daily */}}
{{- end -}}
{{- end -}}

{{/*
Полное доменное имя сервиса (FQDN) с автоматическим определением порта
*/}}
{{- define "serviceFQDN" -}}
{{- $name := required "Service name is required" .name -}}
{{- $namespace := default "default" (coalesce .namespace .Release.Namespace) -}}
{{- $clusterDomain := default "cluster.local" .clusterDomain -}}

{{- /* Базовый FQDN */}}
{{- $fqdn := printf "%s.%s.svc.%s" $name $namespace $clusterDomain -}}

{{- /* Автоматически определяем порт из новой структуры */}}
{{- $port := "" -}}
{{- if hasKey . "Values" -}}
  {{- $serviceValues := index .Values (printf "%s" $name) -}}
  {{- if $serviceValues -}}
    {{- $port = $serviceValues.port -}}
  {{- end -}}
{{- end -}}

{{- /* Добавляем порт если найден */}}
{{- if $port -}}
  {{- printf "%s:%s" $fqdn $port -}}
{{- else -}}
  {{- $fqdn -}}
{{- end -}}
{{- end -}}