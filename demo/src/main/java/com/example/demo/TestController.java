package com.example.demo;


import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController

public class TestController {


    @GetMapping("/api/get")
    public  String  get()
    {
        return "Hello";
    }
}
