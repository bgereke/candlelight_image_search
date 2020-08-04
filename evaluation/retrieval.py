import numpy as np
import evaluation.metrics as metrics


def _retrieve_knn_faiss_gpu_inner_product(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatIP(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return retrieved_result_indices


def _retrieve_knn_faiss_gpu_binary(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexBinaryFlatConfig()
    # flat_config.device = gpu_id
    # index = faiss.GpuIndexBinaryFlat(res, 2048, flat_config)
    # index.add(db_embeddings)
    index = faiss.IndexBinaryFlat(2048)
    index.add(db_embeddings)


    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return retrieved_result_indices


def get_retrieval_results(query_embeddings, db_embeddings, query_labels, db_labels,
                          k=100, gpu_id=0, binary=True):
    retrieval_results = {}

    # get solution dict
    solutions = metrics.get_solution_dict(query_labels, db_labels)

    if binary:
        # ======================== binary embedding evaluation =========================================================
        binary_query_embeddings = np.packbits(np.require(query_embeddings > 0, dtype='uint8'), axis=1)
        binary_db_embeddings = np.packbits(np.require(db_embeddings > 0, dtype='uint8'), axis=1)
        del query_embeddings, db_embeddings

        # knn retrieval from embeddings (binary embeddings, hamming distance)
        retrieved_result_indices = _retrieve_knn_faiss_gpu_binary(binary_query_embeddings,
                                                                  binary_db_embeddings,
                                                                  k,
                                                                  gpu_id=gpu_id)
    else:
        # ======================== float embedding evaluation ==========================================================
        # knn retrieval from embeddings (l2 normalized embedding + inner product = cosine similarity)
        retrieved_result_indices = _retrieve_knn_faiss_gpu_inner_product(query_embeddings,
                                                                         db_embeddings,
                                                                         k,
                                                                         gpu_id=gpu_id)

    # get prediction dict
    predictions = metrics.get_prediction_dict(retrieved_result_indices)

    # evaluate metrics
    retrieval_results['rr_at_k'] = metrics.recall_rate_at_k(retrieved_result_indices,
                                                            query_labels,
                                                            db_labels, k)
    retrieval_results['map'] = metrics.mean_average_precision(predictions, solutions, k)
    retrieval_results['mean_p_k'] = metrics.mean_precisions(predictions, solutions, k)
    mean_position, median_position = metrics.mean_median_position(predictions, solutions, k)
    retrieval_results['mean_position'] = mean_position
    retrieval_results['median_position'] = median_position

    return retrieval_results
