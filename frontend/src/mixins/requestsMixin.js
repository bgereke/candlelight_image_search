const axios = require("axios");
const APIURL = "https://pixabay.com/api";

export const requestsMixin = {
  methods: {
    searchImages(keyword, page = 1) {
      return axios.get(
        `${APIURL}/?page=${page}&key=18137162-8bce742258e73e6063f58d40a&q=${keyword}`
      );
    }
  }
};
