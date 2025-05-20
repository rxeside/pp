#include <iostream>
#include <thread>
#include <vector>
#include <cassert>
#include "TicketOffice.h"

void TestConcurrentSelling()
{
    int totalTickets = 1000;
    TicketOffice office(totalTickets);

    int numThreads = 10;
    int ticketsPerThread = 150;

    std::vector<std::thread> threads;
    std::atomic<int> totalSold{0};

    for (int i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&office, ticketsPerThread, &totalSold]() {
            try
            {
                int sold = office.SellTickets(ticketsPerThread);
                totalSold.fetch_add(sold, std::memory_order_relaxed);
            } catch (const std::exception &ex)
            {
                std::cerr << "Exception: " << ex.what() << std::endl;
            }
        });
    }

    for (auto &thread: threads)
    {
        thread.join();
    }

    int remaining = office.GetTicketsLeft();

    std::cout << "Tickets sold: " << totalSold << std::endl;
    std::cout << "Tickets left: " << remaining << std::endl;

    assert(totalSold + remaining == totalTickets);
}

void TestInvalidInput()
{
    TicketOffice office(100);
    bool exceptionThrown = false;

    try
    {
        office.SellTickets(0);
    } catch (const std::invalid_argument &)
    {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
    assert(office.GetTicketsLeft() == 100);
}

int main()
{
    TestConcurrentSelling();
    TestInvalidInput();
    std::cout << "All tests passed.\n";
    return 0;
}
