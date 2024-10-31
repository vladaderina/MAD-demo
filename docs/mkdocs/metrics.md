## Метрики в Kubernetes
**Метрики** — это измеряемые значения, которые предоставляют информацию о состоянии системы. Например, время выполнения операции, количество запросов, объем используемой памяти и т.д.
В Kubernetes метрики используются для отслеживания различных аспектов работы кластера, таких как производительность API-сервера, состояние узлов, работоспособность подов и т.д. Метрики могут быть использованы для мониторинга, диагностики проблем и принятия решений о масштабировании ресурсов.
Метрики sum измеряют общее время, затраченное на выполнение определенных операций, включая процессорное время, время ожидания ввода-вывода, сетевые задержки и другие задержки. Это позволяет получить полное представление о производительности обработки запросов

## Реестр метрик
**Реестр метрик** — это централизованное хранилище, где регистрируются все метрики. В Kubernetes для этого используется пакет k8s.io/component-base/metrics/legacyregistry.

Реестр позволяет централизованно управлять метриками, регистрировать их и предоставлять доступ к ним через API.

## Что из себя представляют метрики
Рассмотрим пример с метриками Admission Controller.

В документации Kubernetes сказано, что Admissiion Controller выполняют роль  *middleware* в API-сервере: перед обработкой API запроса, например, для создания Pod они добавляют к запросу дополнительную информацию.

Исполнение кода контроллеров занимает процессорное время.

Чтобы не отслеживать 

Метрика **step_admission_duration_seconds**

``` go linenums="1" title="Инициализация метрики"
func newAdmissionMetrics() *AdmissionMetrics {
	controller := &metricSet{
		latencies: metrics.NewHistogramVec(
			&metrics.HistogramOpts{
				Namespace:      namespace,
				Subsystem:      subsystem,
				Name:           "controller_admission_duration_seconds",
				Help:           "Admission controller latency histogram in seconds, identified by name and broken out for each operation and API resource and type (validate or admit).",
				Buckets:        []float64{0.005, 0.025, 0.1, 0.5, 1.0, 2.5},
				StabilityLevel: metrics.STABLE,
			},
			[]string{"name", "type", "operation", "rejected"},
		),

		latenciesSummary: nil,
	}
```

``` go linenums="1" title="Инициализация функции"
func (m *cgroupCommon) Update(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("update").Observe(metrics.SinceInSeconds(start))
	}()
```

## Ключевые характеристики метрик
Типы:

- Histogram - формат предоставляет информацию о распределении времени выполнения. Например, можно узнать, сколько запросов выполняется за 10 мс, 50 мс, 100 мс и т.д.
- Gauge
- Counter
- Custom

Так же для каждой метрики есть понятия:

  - _sum
  - _count
  - _buckets

## Список метрик компонент Control Plane
### Метрики kube-api
1. apiserver_admission_controller_admission_duration_seconds

	Показывает время, затраченное на выполнение контроллеров доступа ([admission controllers](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/)) при обработке запросов к API-серверу.

    - Stability Level: STABLE
	- Type: Histogram
	- Labels: name operation rejected type
    
	**Примеры:**
    
    Допустим мы хотим удалить namespace kube-system, тогда контроллер NamespaceLifecycle запретит это действие, так как kube-system является системным пространством имен.
    
    Или хотим создать под с высокими требованиями CPU, тогда контроллер LimitRanger отклонит создание пода, так как запрошенное количество памяти превышает установленное ограничение.

    И метрика отображает для каждого контроллера:
    - количество запросов, не превышающее время t
    - общее время выполнения
    - общее количество запросов
	
    Пример метрик: (1)
	{ .annotate }

    1.  : apiserver_admission_controller_admission_duration_seconds_bucket{name="ValidatingAdmissionWebhook",operation="UPDATE",rejected="false",type="validate",le="2.5"} 49691
    	apiserver_admission_controller_admission_duration_seconds_bucket{name="ValidatingAdmissionWebhook",operation="UPDATE",rejected="false",type="validate",le="+Inf"} 49691
    	apiserver_admission_controller_admission_duration_seconds_sum{name="ValidatingAdmissionWebhook",operation="UPDATE",rejected="false",type="validate"} 0.16454242599999996
    	apiserver_admission_controller_admission_duration_seconds_count{name="ValidatingAdmissionWebhook",operation="UPDATE",rejected="false",type="validate"} 49691

2. apiserver_admission_step_admission_duration_seconds

	Показывает распределение времени выполнения шагов допуска (admission steps) в API-сервере.
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels: operation rejected type

    Пример метрик:

    apiserver_admission_step_admission_duration_seconds_summary{operation="UPDATE",rejected="false",type="validate",quantile="0.99"} 5.6764e-05
    apiserver_admission_step_admission_duration_seconds_summary_sum{operation="UPDATE",rejected="false",type="validate"} 1.269520069000019
    apiserver_admission_step_admission_duration_seconds_summary_count{operation="UPDATE",rejected="false",type="validate"} 50556

3. apiserver_admission_webhook_admission_duration_seconds
	
	Показывает распределение времени выполнения вебхуков доступа (admission webhooks) в API-сервере.
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels: nameo peration rejected type

    Пример метрик: - (по умолчанию в кубере нет вебхуков доступа)


4. apiserver_current_inflight_requests

	Предоставляет информацию о максимальном количестве одновременно обрабатываемых запросов (inflight requests) в течение последней секунды для каждого типа запроса (request kind).
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels: request_kind

    Пример метрик:

    apiserver_current_inflight_requests{request_kind="mutating"} 0
    apiserver_current_inflight_requests{request_kind="readOnly"} 1

5. apiserver_longrunning_requests

    Предоставляет количество всех активных долговременных запросов (long-running requests), которые в данный момент обрабатываются API Server. Эта метрика разбивается по различным атрибутам, таким как тип операции (verb), группа (group), версия (version), ресурс (resource), область (scope) и компонент (component).
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels: component group resource scope subresource verb version
    
    Пример метрик:
    
    apiserver_longrunning_requests{component="apiserver",group="storage.k8s.io",resource="storageclasses",scope="cluster",subresource="",verb="WATCH",version="v1"} 4
    apiserver_longrunning_requests{component="apiserver",group="",resource="configmaps",scope="cluster",subresource="",verb="WATCH",version="v1"} 1
    apiserver_longrunning_requests{component="apiserver",group="",resource="configmaps",scope="namespace",subresource="",verb="WATCH",version="v1"} 1
    apiserver_longrunning_requests{component="apiserver",group="",resource="configmaps",scope="resource",subresource="",verb="WATCH",version="v1"} 6

6. apiserver_request_duration_seconds

	Предоставляет информацию о распределении задержки (latency) ответов в секундах для каждого типа операции (verb), значения (dry_run), группы (group), версии (version), ресурса (resource), подресурса (subresource), области (scope) и компонента (component).
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels: component dry_run group resource scope subresource verb version

7. apiserver_request_total

	Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels: code component dry_run group resource scope subresource verb version

8. apiserver_requested_deprecated_apis

	Gauge of deprecated APIs that have been requested, broken out by API group, version, resource, subresource, and removed_release.
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels:groupremoved_releaseresourcesubresourceversion

9. apiserver_response_sizes

	Response size distribution in bytes for each group, version, verb, resource, subresource, scope and component.
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels:componentgroupresourcescopesubresourceverbversion

10.  apiserver_storage_objects

	Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels:resource

11. apiserver_storage_size_bytes

	Size of the storage database file physically allocated in bytes.
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:storage_cluster_id

### Метрики scheduler

1. scheduler_framework_extension_point_duration_seconds

	Latency for running all plugins of a specific extension point.
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels:extension_pointprofilestatus

2. scheduler_pending_pods

	Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulablePods that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels:queue

3. scheduler_pod_scheduling_attempts

	Number of attempts to successfully schedule a pod.
	
	- Stability Level:STABLE
	- Type: Histogram

4. scheduler_pod_scheduling_duration_seconds

	E2e latency for a pod being scheduled which may include multiple scheduling attempts.
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels:attempts
	- Deprecated Versions:1.29.0

5. scheduler_preemption_attempts_total

	Total preemption attempts in the cluster till now
	
	- Stability Level:STABLE
	- Type: Counter

6. scheduler_preemption_victims

	Number of selected preemption victims
	
	- Stability Level:STABLE
	- Type: Histogram

7. scheduler_queue_incoming_pods_total

	Number of pods added to scheduling queues by event and queue type.
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:eventqueue

8. scheduler_schedule_attempts_total

	Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:profileresult

9. scheduler_scheduling_attempt_duration_seconds

	Scheduling attempt latency in seconds (scheduling algorithm + binding)
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels:profileresult

### Метрики controller manager

1. cronjob_controller_job_creation_skew_duration_seconds

	Time between when a cronjob is scheduled to be run, and when the corresponding job is created
	
	- Stability Level:STABLE
	- Type: Histogram

2. job_controller_job_pods_finished_total

	The number of finished Pods that are fully tracked
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:completion_moderesult

3. job_controller_job_sync_duration_seconds

	The time it took to sync a job
	
	- Stability Level:STABLE
	- Type: Histogram
	- Labels:actioncompletion_moderesult

4. job_controller_job_syncs_total

	The number of job syncs
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:actioncompletion_moderesult

5. job_controller_jobs_finished_total

	The number of finished jobs
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:completion_modereasonresult

### Метрики node
1. node_collector_evictions_total

	Number of Node evictions that happened since current instance of NodeController started.
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:zone

2. node_cpu_usage_seconds_total

	Cumulative cpu time consumed by the node in core-seconds
	
	- Stability Level:STABLE
	- Type: Custom

3. node_memory_working_set_bytes

	Current working set of the node in bytes
	
	- Stability Level:STABLE
	- Type: Custom

4. pod_cpu_usage_seconds_total

	Cumulative cpu time consumed by the pod in core-seconds
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:podnamespace

5. pod_memory_working_set_bytes

	Current working set of the pod in bytes
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:podnamespace

6. resource_scrape_error

	1 if there was an error while getting container metrics, 0 otherwise
	
	- Stability Level:STABLE
	- Type: Custom

### Метрики pod
1. kube_pod_resource_limit

	Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:namespacepodnodeschedulerpriorityresourceunit

2. kube_pod_resource_request

	Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:namespacepodnodeschedulerpriorityresourceunit

3. kubernetes_healthcheck

	This metric records the result of a single healthcheck.
	
	- Stability Level:STABLE
	- Type: Gauge
	- Labels:nametype

4. kubernetes_healthchecks_total

	This metric records the results of all healthcheck.
	
	- Stability Level:STABLE
	- Type: Counter
	- Labels:namestatustype

### Метрики container
1. container_cpu_usage_seconds_total

	Cumulative cpu time consumed by the container in core-seconds
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:containerpodnamespace

2. container_memory_working_set_bytes

	Current working set of the container in bytes
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:containerpodnamespace

3. container_start_time_seconds

	Start time of the container since unix epoch in seconds
	
	- Stability Level:STABLE
	- Type: Custom
	- Labels:containerpodnamespace

### Метрики etcd
1. etcd_bookmark_counts

	Number of etcd bookmarks (progress notify events) split by kind.
	
	- Stability Level:ALPHA
	- Type: Gauge
	- Labels:resource

2. etcd_lease_object_counts

	Number of objects attached to a single etcd lease.
	
	- Stability Level:ALPHA
	- Type: Histogram

3. etcd_request_duration_seconds

	Etcd request latency in seconds for each operation and object type.
	
	- Stability Level:ALPHA
	- Type: Histogram
	- Labels:operationtype

4. etcd_request_errors_total

	Etcd failed request counts for each operation and object type.
	
	- Stability Level:ALPHA
	- Type: Counter
	- Labels:operationtype

5. etcd_requests_total

	Etcd request counts for each operation and object type.
	
	- Stability Level:ALPHA
	- Type: Counter
	- Labels:operationtype

6. etcd_version_info

	Etcd server's binary version
	
	- Stability Level:ALPHA
	- Type: Gauge
	- Labels:binary_version


## Полезные команды
- Команда по отображению всех метрик
``` bash linenums="1"
    kubectl get --raw /metrics
```
- Список включенных плагинов допуска
``` bash linenums="1"
kubectl get pod -n kube-system -l component=kube-apiserver -o yaml | grep enable-admission-plugins
```
Пример вывода команды (плагины допуска по умолчанию):
```
- --enable-admission-plugins = NamespaceLifecycle, LimitRanger, ServiceAccount, DefaultStorageClass, DefaultTolerationSeconds, NodeRestriction, MutatingAdmissionWebhook, ValidatingAdmissionWebhook, ResourceQuota
```